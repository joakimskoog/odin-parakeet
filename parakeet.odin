package parakeet

import "core:c"

when ODIN_OS == .Windows {
	foreign import lib "lib/win-x64-cpu/parakeet.lib"
} else when ODIN_OS == .Darwin {
	foreign import lib "lib/macos-x64-cpu/libparakeet.dylib"
} else when ODIN_OS == .Linux {
	when ODIN_ARCH == .amd64 {
		foreign import lib "lib/linux-x64-cpu/libparakeet.so"
	} else when ODIN_ARCH == .arm64 {
		foreign import lib "lib/linux-arm64-cpu/libparakeet.so"
	}
}

// Odin bindings for the parakeet.cpp C API (https://github.com/mudler/parakeet.cpp)
//
// A context represents a loaded Parakeet model and its associated state.
// It is used for all transcription operations and must be properly initialized and released with free(ctx).
// Strings returned by transcription and streaming procedures are allocated by the library and must be released with free_string(transcript).

// C API version targeted by these bindings. The binding will panic if there's a mismatch.
BINDINGS_VERSION :: 6
BINDINGS_VERSION_STRING :: "6"

// Owning model context returned by load. Must be released with free(ctx).
Context :: distinct rawptr

// Selects which decoder head is used for transcription
Decoder :: enum c.int {
	Default    = 0, // Select CTC or the transducer according to the model's architecture
	CTC        = 1, // Force the CTC decoder head
	Transducer = 2, // Force the TDT/RNN-T transducer head
}

// Streaming session created from a Context.
//
// The originating Context must remain alive until the stream is released with stream_free.
Stream :: distinct rawptr

// Values that may be present in the event mask returned by stream_feed.
EVENT_EOU :: c.int(1) // The speaker completed an utterance
EVENT_EOB :: c.int(2) // The speaker produced a backchannel acknowledgement.

// An EOU or EOB event emitted by the streaming decoder.
Stream_Event :: struct {
	token:         c.int, // Raw vocabulary ID of the special token.
	is_eob:        c.int, // 0 for EOU; 1 for EOB.
	encoder_frame: c.int, // Absolute encoder-output frame.
	time_sec:      f32, // Time from the start of the stream, in seconds.
}

#assert(size_of(Stream_Event) == 16)

@(init)
version_check :: proc "contextless" () {
	actual_version := abi_version()
	if actual_version != BINDINGS_VERSION {
		buf: [256]byte
		n := copy(buf[:], "Parakeet version mismatch: ")
		n += copy(buf[n:], "bindings are for version ")
		n += copy(buf[n:], BINDINGS_VERSION_STRING)
		n += copy(buf[n:], ", make sure to link the correct parakeet.cpp version ")

		panic_contextless(string(buf[:n]))
	}
}

@(default_calling_convention = "c", link_prefix = "parakeet_capi_")
foreign lib {

	// ABI version of this header/implementation. Bump on any breaking change to the
	// function signatures or semantics below.
	//
	// v3: added the target_lang variants (parakeet_capi_transcribe_path_lang,
	//     parakeet_capi_transcribe_pcm_lang, parakeet_capi_stream_begin_lang,
	//     parakeet_capi_transcribe_pcm_batch_json_lang,
	//     parakeet_capi_transcribe_pcm_batch_lang) for multilingual
	//     prompt-conditioned (nemotron) models. The original non-lang entry points
	//     are unchanged and delegate with the model default language.
	//
	// v4: added the streaming JSON entry points (parakeet_capi_stream_feed_json,
	//     parakeet_capi_stream_finalize_json) that surface per-word timestamps
	//     (start/end/conf) plus frame_sec alongside the newly-finalized text, and
	//     added "frame_sec" to the transcribe_*_json documents. The original entry
	//     points are unchanged.
	//
	// v5: the <EOU> (end of utterance) vs <EOB> (end of backchannel) distinction is
	//     now visible across the C boundary. BREAKING semantics on the streaming
	//     surface: parakeet_capi_stream_feed's `*eou_out` is now a bitmask
	//     (PARAKEET_EVENT_EOU | PARAKEET_EVENT_EOB) instead of an any-event 0/1,
	//     and the JSON "eou" field now means "an <EOU> fired" only, with a new
	//     "eob" field beside it (in v4 both meant "an <EOU> OR <EOB> fired").
	//     Added parakeet_capi_stream_drain_events (typed per-event records with
	//     is_eob + timestamps, freed with parakeet_capi_free_events) and an
	//     "events" array in the stream_feed_json / stream_finalize_json documents.
	//
	// v6: added parakeet_capi_transcribe_pcm_logits, exposing the CTC head's
	//     log-prob matrix (row-major [T, vocab+1], already log-softmaxed) instead
	//     of decoded text — for external LM/decoder stacks (e.g. pyctcdecode +
	//     KenLM) that need the raw distribution rather than this library's own
	//     greedy/beam decode. Freed with the new parakeet_capi_free_logits. The
	//     original entry points are unchanged.
	abi_version :: proc() -> c.int ---


	// Loads a GGUF model and returns its owning context.
	//
	// Returns nil on failure. A successful context must be released with free.
	load :: proc(gguf_path: cstring) -> Context ---

	// Releases a context returned by load. Safe to call with nil.
	free :: proc(ctx: Context) ---

	// Transcribes a WAV file.
	//
	// The returned UTF-8 string is owned by the caller and must be released with
	// free_string. Returns nil on failure; call last_error for details.
	transcribe_path :: proc(ctx: Context, wav_path: cstring, decoder: Decoder) -> cstring ---

	// Like transcribe_path, but selects a language prompt.
	//
	// target_lang may be a locale such as "en", "de", or "auto". Passing nil or an
	// empty string uses the model default. The value is ignored by models that do
	// not support language prompts.
	transcribe_path_lang :: proc(ctx: Context, wav_path: cstring, decoder: Decoder, target_lang: cstring) -> cstring ---

	// Transcribes mono floating-point PCM.
	//
	// samples points to n_samples normalized f32 samples. Audio whose sample rate
	// is not 16 kHz is resampled by the library.
	//
	// The returned UTF-8 string must be released with free_string. Returns nil on
	// failure; call last_error for details.
	transcribe_pcm :: proc(ctx: Context, samples: [^]f32, n_samples: c.int, sample_rate: c.int, decoder: Decoder) -> cstring ---

	// Like transcribe_pcm but selects the language prompt (see
	// transcribe_path_lang for `target_lang` semantics).
	transcribe_pcm_lang :: proc(ctx: Context, samples: [^]f32, n_samples: c.int, sample_rate: c.int, decoder: Decoder, target_lang: cstring) -> cstring ---

	// Transcribes multiple PCM clips.
	//
	// samples and n_samples each contain n_clips entries. out must contain room for
	// n_clips strings.
	//
	// On success, returns zero and fills out with strings owned by the caller.
	// Release every returned string with free_string.
	//
	// On failure, returns nonzero and leaves all out entries nil.
	transcribe_pcm_batch :: proc(ctx: Context, samples: [^][^]f32, n_samples: [^]c.int, n_clips: c.int, sample_rate: c.int, decoder: Decoder, out: [^]cstring) -> c.int ---

	// Like transcribe_pcm_batch but selects one language for the whole batch.
	transcribe_pcm_batch_lang :: proc(ctx: Context, samples: [^][^]f32, n_samples: [^]c.int, n_clips: c.int, sample_rate: c.int, decoder: Decoder, target_lang: cstring, out: [^]cstring) -> c.int ---

	// Transcribes a WAV file and returns timestamps and confidence as JSON.
	//
	// The document has the following shape:
	//
	// {
	//   "text": "...",
	//   "frame_sec": 0.08,
	//   "words":  [{"w": "...", "start": 0.48, "end": 0.64, "conf": 0.91}],
	//   "tokens": [{"id": 123, "t": 0.48, "conf": 0.91}]
	// }
	//
	// Times are measured in seconds. The returned string must be released with
	// free_string.
	transcribe_path_json :: proc(ctx: Context, wav_path: cstring, decoder: Decoder) -> cstring ---

	// Transcribes concatenated PCM clips and returns one JSON array.
	//
	// n_samples contains the length of each clip. The sum of its n_clips entries
	// must exactly equal the number of samples available through samples_concat.
	// Violating this precondition may cause an out-of-bounds read.
	//
	// The returned string must be released with free_string.
	transcribe_pcm_batch_json :: proc(ctx: Context, samples_concat: [^]f32, n_samples: [^]c.int, n_clips: c.int, sample_rate: c.int, decoder: Decoder) -> cstring ---

	// Like transcribe_pcm_batch_json but selects one language for the batch.
	transcribe_pcm_batch_json_lang :: proc(ctx: Context, samples_concat: [^]f32, n_samples: [^]c.int, n_clips: c.int, sample_rate: c.int, decoder: Decoder, target_lang: cstring) -> cstring ---

	// Runs offline TDT beam search and returns ranked hypotheses as JSON.
	//
	// beam_size must be greater than or equal to nbest, and nbest must be at least
	// one. A nonzero score_norm enables sequence-length-normalized ranking.
	//
	// The returned string must be released with free_string.
	transcribe_path_nbest_json :: proc(ctx: Context, wav_path: cstring, beam_size: c.int, nbest: c.int, score_norm: c.int, target_lang: cstring) -> cstring ---

	// Runs offline TDT beam search on PCM and returns ranked hypotheses as JSON.
	//
	// beam_size must be greater than or equal to nbest, and nbest must be at least
	// one. A nonzero score_norm enables sequence-length-normalized ranking.
	//
	// The returned string must be released with free_string.
	transcribe_pcm_nbest_json :: proc(ctx: Context, samples: [^]f32, n_samples: c.int, sample_rate: c.int, beam_size: c.int, nbest: c.int, score_norm: c.int, target_lang: cstring) -> cstring ---

	// Runs the mel frontend, encoder, and CTC head without decoding the result.
	//
	// On success, returns zero and sets out_logits to a row-major [T, vocab+1]
	// matrix of log-softmaxed f32 values:
	//
	//     logits[t*out_vocab_plus_1 + vocabulary_index]
	//
	// Release the returned buffer with free_logits.
	//
	// On failure, returns nonzero. When all out-parameters are valid, out_logits is
	// left nil and both dimensions are set to zero.
	transcribe_pcm_logits :: proc(ctx: Context, samples: [^]f32, n_samples: c.int, sample_rate: c.int, out_logits: ^[^]f32, out_t: ^c.int, out_vocab_plus_1: ^c.int) -> c.int ---

	// Releases a logits buffer returned by transcribe_pcm_logits. Safe with nil.
	free_logits :: proc(logits: [^]f32) ---

	// Begins a streaming session using the model loaded in ctx.
	//
	// The context must remain alive until the returned stream is released. Returns
	// nil if the model does not support streaming or initialization fails.
	stream_begin :: proc(ctx: Context) -> Stream ---

	// Begins a streaming session with a language prompt.
	//
	// target_lang may be a locale such as "en", "de", or "auto". Passing nil or
	// an empty string uses the model default. The value is ignored by models
	// without language prompts.
	//
	// Returns nil on failure; call last_error on ctx for details.
	stream_begin_lang :: proc(ctx: Context, target_lang: cstring) -> Stream ---

	// Feeds 16 kHz mono PCM into a streaming session.
	//
	// Returns the text finalized since the previous call. An empty, non-nil string
	// means that no text was finalized. A nil result indicates an error.
	//
	// When event_mask_out is non-nil, it receives zero or a bitwise combination of
	// EVENT_EOU and EVENT_EOB.
	//
	// Release the returned string with free_string.
	stream_feed :: proc(stream: Stream, pcm: [^]f32, n_samples: c.int, event_mask_out: ^c.int) -> cstring ---

	// Flushes buffered audio and returns any remaining finalized text.
	//
	// An empty, non-nil string means no additional text was finalized. A nil
	// result indicates an error. Finalizing does not fabricate an EOU event.
	// Release the returned string with free_string.
	stream_finalize :: proc(stream: Stream) -> cstring ---

	// Drains EOU and EOB events accumulated since the previous drain.
	//
	// On success, returns the number of events and sets out_events to an allocated
	// array when the count is greater than zero. Release that array with
	// free_events. When no events are available, returns zero and sets out_events
	// to nil.
	//
	// Returns -1 for an invalid stream or output pointer.
	//
	// The event queue is shared with stream_feed_json and stream_finalize_json,
	// which also drain it. Use either the typed-event API or the JSON API for a
	// given stream.
	stream_drain_events :: proc(stream: Stream, out_events: ^[^]Stream_Event) -> c.int ---

	// Releases an event array returned by stream_drain_events. Safe with nil.
	free_events :: proc(events: [^]Stream_Event) ---

	// Feeds streaming PCM and returns finalized text, events, words, and
	// timestamps as JSON. This also drains the stream's event queue.
	//
	// Returns nil on failure. Release a successful result with free_string.
	stream_feed_json :: proc(stream: Stream, pcm: [^]f32, n_samples: c.int) -> cstring ---

	// Flush the final streaming tail and return the result as JSON.
	stream_finalize_json :: proc(stream: Stream) -> cstring ---

	// Releases a streaming session. Safe to call with nil.
	stream_free :: proc(stream: Stream) ---

	// Releases a string returned by a transcription or streaming procedure.
	// Safe to call with nil.
	free_string :: proc(s: cstring) ---

	// Returns the most recent error associated with ctx.
	//
	// The returned string is borrowed from the context and must not be freed. It
	// remains valid only until the next call using that context or until the
	// context is released. Returns an empty string when ctx is nil.
	last_error :: proc(ctx: Context) -> cstring ---
}
