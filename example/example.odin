package parakeet_example

import parakeet "../"
import "core:c"
import "core:fmt"


main :: proc() {
	version := parakeet.abi_version()
	fmt.println("Parakeet ABI version: ", version)

	//Replace this with path to your model file
	ctx := parakeet.load("tdt_ctc-110m-q4_k.gguf")
	defer parakeet.free(ctx)

	transcript := parakeet.transcribe_path(ctx, "speech.wav", .Default)
	defer parakeet.free_string(transcript)
	fmt.println(transcript)

	parakeet.transcribe_path(ctx, "Force_Error.wav", .Default)
	fmt.println("Last error: ", parakeet.last_error(ctx))

	pcm_transcript := parakeet.transcribe_pcm(
		ctx,
		raw_data(SPEECH_SAMPLES[:]),
		c.int(len(SPEECH_SAMPLES)),
		SPEECH_SAMPLE_RATE,
		.Default,
	)
	defer parakeet.free_string(pcm_transcript)
	fmt.println(pcm_transcript)

}
