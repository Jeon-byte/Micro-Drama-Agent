# Run with flags
# # Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
# f5-tts_infer-cli --model "F5-TTS" --ref_audio "ref_audio.wav" --ref_text "The content, subtitle or transcription of reference audio." --gen_text "Some text you want TTS model generate for you."

# # Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
# f5-tts_infer-cli
# # Or with your own .toml file
# f5-tts_infer-cli -c custom.toml

# # Multi voice. See src/f5_tts/infer/README.md
# f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml