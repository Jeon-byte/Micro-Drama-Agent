from utils.prompt_making import make_prompt
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# ### Use given transcript
# make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav",
#                 transcript="Just, what was that? Paimon thought we were gonna get eaten.")

### Alternatively, use whisper
make_prompt(name="Rui", audio_prompt_path="en-2.wav")


# download and load all models
preload_models()

text_prompt = """
I agree, let's explore some real-world applications next
"""
audio_array = generate_audio(text_prompt, prompt="Rui")

write_wav("Rui.wav", SAMPLE_RATE, audio_array)