from configparser import ConfigParser
import torch


class ToontraOptions(ConfigParser):
    def __init__(self, defaults=None, config_file="Toontra_new/config/config.ini"):
        ConfigParser.__init__(self, defaults=defaults)
        config = ConfigParser()
        config.read(config_file)
        self._config = config
        self.config_file = config_file
        self.num_class = None

    # TEXTDETECTOR
    @property
    def TextDetector_weights_dir(self):
        return self._config.get("TEXTDETECTOR", "weights_dir")

    # TEXTRECOGNIZER
    @property
    def saved_model(self):
        return self._config.get("TEXTRECOGNIZER", "saved_model")

    @property
    def batch_max_length(self):
        return self._config.getint("TEXTRECOGNIZER", "batch_max_length")

    @property
    def imgH(self):
        return self._config.getint("TEXTRECOGNIZER", "imgH")

    @property
    def imgW(self):
        return self._config.getint("TEXTRECOGNIZER", "imgW")

    @property
    def PAD(self):
        return self._config.getboolean("TEXTRECOGNIZER", "PAD")

    @property
    def rgb(self):
        return self._config.getboolean("TEXTRECOGNIZER", "rgb")

    @property
    def Transformation(self):
        return self._config.get("TEXTRECOGNIZER", "Transformation")

    @property
    def FeatureExtraction(self):
        return self._config.get("TEXTRECOGNIZER", "FeatureExtraction")

    @property
    def SequenceModeling(self):
        return self._config.get("TEXTRECOGNIZER", "SequenceModeling")

    @property
    def Prediction(self):
        return self._config.get("TEXTRECOGNIZER", "Prediction")

    @property
    def num_fiducial(self):
        return self._config.getint("TEXTRECOGNIZER", "num_fiducial")

    @property
    def input_channel(self):
        return self._config.getint("TEXTRECOGNIZER", "input_channel")

    @property
    def output_channel(self):
        return self._config.getint("TEXTRECOGNIZER", "output_channel")

    @property
    def hidden_size(self):
        return self._config.getint("TEXTRECOGNIZER", "hidden_size")

    @property
    def batch_size(self):
        return self._config.getint("TEXTRECOGNIZER", "batch_size")

    # TEXTTRANSLATOR
    @property
    def TextTranslator_papago_client_id(self):
        return self._config.get("TEXTTRANSLATOR", "papago_client_id")

    @property
    def TextTranslator_papago_client_secret(self):
        return self._config.get("TEXTTRANSLATOR", "papago_client_secret")

    @property
    def TextTranslator_papago_url(self):
        return self._config.get("TEXTTRANSLATOR", "papago_url")

    @property
    def TextTranslator_google_credentials(self):
        return self._config.get("TEXTTRANSLATOR", "google_credentials")

    @property
    def TextTranslator_deepl_auth_key(self):
        return self._config.get("TEXTTRANSLATOR", "deepl_auth_key")

    @property
    def TextTranslator_openai_api_key(self):
        return self._config.get("TEXTTRANSLATOR", "openai_api_key")

    # IMAGEINPAINTER
    @property
    def ImageInpainter_weights(self):
        return self._config.get("IMAGEINPAINTER", "weights")

    # Toontra_new
    @property
    def cuda(self):
        return self._config.getboolean("Toontra_new", "cuda")

    @property
    def gpu_id(self):
        return self._config.getint("Toontra_new", "gpu_id")

    @property
    def log_dir(self):
        return self._config.get("Toontra_new", "log_dir")

    @property
    def device(self):
        # return torch.device("cuda" if self.cuda else "cpu")
        use_cuda = self.cuda and torch.cuda.is_available()
        return torch.device("cuda" if use_cuda else "cpu")


if __name__ == "__main__":
    config = ToontraOptions()
    print("=" * 100)
    print(config.config_file)
