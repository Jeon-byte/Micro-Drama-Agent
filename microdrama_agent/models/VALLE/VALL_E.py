from PIL import Image, ImageDraw, ImageFont

from .utils.prompt_making import make_prompt
from .utils.generation import generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import os


class VALLE_pipe:
    def __init__(self, photo_audio_path,characters_list):
        self.SAMPLE_RATE = 24000
        # download and load all models
        preload_models()

        import os

        for name in characters_list:
            print("Create a prompt for audio of character {} in the VALLE model.".format(name))
            
            audio_path = os.path.join(photo_audio_path,name,name+".wav")
            if not os.path.exists(audio_path):
                print("audio of character {} does not exist in the VALLE model".format(name))
                continue
            make_prompt(name=name, audio_prompt_path=audio_path)


    def predict(self, subtitle, save_path):
        
        
        for i,name in enumerate(subtitle):
            save_path_one = save_path.replace(".jpg","")+"_"+str(i)+"_"+name+".wav"
            text_prompt = subtitle[name]
            audio_array = generate_audio(text_prompt, prompt=name)

            write_wav(save_path_one, self.SAMPLE_RATE, audio_array)
