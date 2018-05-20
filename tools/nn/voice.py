import torchaudio




def voice_train():
    sound, sample_rate = torchaudio.load('foo.mp3')
    torchaudio.save('foo.mp3.pt', sound, sample_rate) # saves tensor to filepass


if __name__ == '__main__':
    voice_train()