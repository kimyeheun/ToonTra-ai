# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, List, Optional, Type
import urllib.request
import urllib.parse
from google.cloud import translate_v2 as translate
from ocr_service.utils.util import array1d_to_array2d, array2d_to_array1d
import deepl
import logging
from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger(__name__)

from config.language_support.language_supports import LANGUAGE_SUPPORTS

GOOGLE_LANGUAGES = {
    "af": "afrikaans",
    "sq": "albanian",
    "am": "amharic",
    "ar": "arabic",
    "hy": "armenian",
    "az": "azerbaijani",
    "eu": "basque",
    "be": "belarusian",
    "bn": "bengali",
    "bs": "bosnian",
    "bg": "bulgarian",
    "ca": "catalan",
    "ceb": "cebuano",
    "ny": "chichewa",
    "zh-cn": "chinese",
    "zh-tw": "chinese_traditional",
    "co": "corsican",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "eo": "esperanto",
    "et": "estonian",
    "tl": "filipino",
    "fi": "finnish",
    "fr": "french",
    "fy": "frisian",
    "gl": "galician",
    "ka": "georgian",
    "de": "german",
    "el": "greek",
    "gu": "gujarati",
    "ht": "haitian_creole",
    "ha": "hausa",
    "haw": "hawaiian",
    "iw": "hebrew",
    "he": "hebrew",
    "hi": "hindi",
    "hmn": "hmong",
    "hu": "hungarian",
    "is": "icelandic",
    "ig": "igbo",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "ja": "japanese",
    "jw": "javanese",
    "kn": "kannada",
    "kk": "kazakh",
    "km": "khmer",
    "ko": "korean",
    "ku": "kurdish",
    "ky": "kyrgyz",
    "lo": "lao",
    "la": "latin",
    "lv": "latvian",
    "lt": "lithuanian",
    "lb": "luxembourgish",
    "mk": "macedonian",
    "mg": "malagasy",
    "ms": "malay",
    "ml": "malayalam",
    "mt": "maltese",
    "mi": "maori",
    "mr": "marathi",
    "mn": "mongolian",
    "my": "myanmar",
    "ne": "nepali",
    "no": "norwegian",
    "or": "odia",
    "ps": "pashto",
    "fa": "persian",
    "pl": "polish",
    "pt": "portuguese",
    "pa": "punjabi",
    "ro": "romanian",
    "sm": "samoan",
    "gd": "scots_gaelic",
    "sr": "serbian",
    "st": "sesotho",
    "sn": "shona",
    "sd": "sindhi",
    "si": "sinhala",
    "sk": "slovak",
    "sl": "slovenian",
    "so": "somali",
    "es": "spanish",
    "su": "sundanese",
    "sw": "swahili",
    "sv": "swedish",
    "tg": "tajik",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "ug": "uyghur",
    "uz": "uzbek",
    "vi": "vietnamese",
    "cy": "welsh",
    "xh": "xhosa",
    "yi": "yiddish",
    "yo": "yoruba",
    "zu": "zulu",
}

GOOGLE_LANGUAGES_CODE = {v.capitalize(): k for k, v in GOOGLE_LANGUAGES.items()}

PAPAGO_LANGUAGES = {
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Chinese_traditional": "zh-TW",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Thai": "th",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "French": "fr",
    "Korean": "ko",
}

DEEPL_LANGUAGES = {
    "Bulgarian": "BG",
    "Czech": "CS",
    "Danish": "DA",
    "German": "DE",
    "Greek": "EL",
    "English": "EN-US",
    "English_UK": "EN-GB",
    "Spanish": "ES",
    "Estonian": "ET",
    "Finnish": "FI",
    "French": "FR",
    "Hungarian": "HU",
    "Indonesian": "ID",
    "Italian": "IT",
    "Japanese": "JA",
    "Korean": "KO",
    "Lithuanian": "LT",
    "Latvian": "LV",
    "Norwegian": "NB",
    "Dutch": "NL",
    "Polish": "PL",
    "Portuguese": "PT-BR",  # PT-BR or PT-PT
    "Romanian": "RO",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Swedish": "SV",
    "Turkish": "TR",
    "Ukrainian": "UK",
    "Chinese": "ZH",
}


class Translaters:
    def __init__(self, opt):
        self.opt = opt

    TRNSLATORLIST = {
        "GOOGLE": [
            "google",
            "Google",
            "GOOGLE",
            "Google Translate",
            "Google Translate API",
        ],
        "PAPAGO": ["papago", "Papago", "PAPAGO", "Papago API", "Naver Papago API"],
        "DEEPL": ["deepl", "DEEPL", "DeepL", "DeepL API", "DeepL API v2"],
        "BAIDU": [
            "baidu",
            "Baidu",
            "BAIDU",
            "Baidu API",
            "Baidu Translate API",
            "Baidu Translate API v2",
        ],
        "OPENAI": ["openai", "OpenAI", "OPENAI", "OpenAI API", "OpenAI API v2"],
    }

    def get_translator(self, translator_name):
        if translator_name in self.TRNSLATORLIST["GOOGLE"]:
            return GoogleTranslater(self.opt)
        elif translator_name in self.TRNSLATORLIST["PAPAGO"]:
            return PapagoTranslater(self.opt)
        elif translator_name in self.TRNSLATORLIST["DEEPL"]:
            return DeepLTranslater(self.opt)
        elif translator_name in self.TRNSLATORLIST["BAIDU"]:
            return BaiduTranslater(self.opt)
        elif translator_name in self.TRNSLATORLIST["OPENAI"]:
            return OpenAITranslater(self.opt)
        else:
            raise ValueError(f"Invalid translator name: {translator_name}")

    def translate(
        self,
        text_list_group,
        target_language,
        source_language,
        translator_names=["GOOGLE"],
        many=False,
    ) -> Dict[str, List[List[str]]]:
        # Normalize input to avoid slicing single string like "deepl" -> 'd'
        if translator_names is None:
            translator_names = ["GOOGLE"]
        if isinstance(translator_names, str):
            translator_names = [translator_names]
        if len(translator_names) > 1:
            return self.translate_with_multiple_models(
                text_list_group, target_language, source_language, translator_names
            )
        else:
            return self.translate_with_one_model(
                text_list_group, target_language, source_language, translator_names[0]
            )

    def translate_with_one_model(
        self,
        text_list_group,
        target_language,
        source_language,
        translator_name="GOOGLE",
    ) -> Dict[str, List[List[str]]]:
        # 2D -> 1D(flat) with shape keep
        flat_list, shape = array2d_to_array1d(text_list_group)
        expected_len = len(flat_list)
        joined_text = "\n".join(flat_list)

        translator = self.get_translator(translator_name)

        try:
            translated_text_in_flat_array = translator.translate(
                joined_text,
                self._convert_language_code(target_language, translator_name),
                self._convert_language_code(source_language, translator_name),
            )
        except Exception:
            translated_text_in_flat_array = "\n".join([""] * expected_len)

        # restore shape
        parts_array_1d = str(translated_text_in_flat_array).split("\n")[:expected_len]

        logger.info(f"expected_len: {expected_len}\n")
        logger.info(f"parts_array_1d: {parts_array_1d}\n")
        logger.info(f"parts_array_1d len: {len(parts_array_1d)}\n")
        logger.info(f"translated_text_in_flat_array: {translated_text_in_flat_array}\n")
        logger.info(
            f"translated_text_in_flat_array len: {len(translated_text_in_flat_array)}\n"
        )
        logger.info(f"shape: {shape}\n")
        logger.info(f"shape len: {len(shape)}\n")
        logger.info(f"parts_array_1d len: {len(parts_array_1d)}\n")
        restored = array1d_to_array2d(parts_array_1d, shape)
        return {str(translator_name).lower(): restored}

    def translate_with_multiple_models(
        self,
        text_list_group: List[List[str]],
        target_language: str,
        source_language: str,
        translator_names: List[str] = None,
    ) -> Dict[str, List[List[str]]]:

        if translator_names is None:
            translator_names = ["GOOGLE"]

        # 단일 엔진명이 문자열로 전달된 경우 리스트로 변환
        if isinstance(translator_names, str):
            translator_names = [translator_names]

        results = {}
        for engine_name in translator_names:
            engine_result = self.translate_with_one_model(
                text_list_group, target_language, source_language, engine_name
            )
            results.update(engine_result)

        return results

    @staticmethod
    def _convert_language_code(language_name, translator_name):
        if translator_name == "GOOGLE":
            # 언어 코드를 언어 이름으로 변환 (ko → korean)
            language_code = language_name.lower()
            return LANGUAGE_SUPPORTS["GOOGLE"].get(language_code, language_name)
        else:
            return language_name


class GoogleTranslater:
    def __init__(self, opt):
        self.opt = opt
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            self.opt.TextTranslator_google_credentials
        )
        self.lang_code = GOOGLE_LANGUAGES_CODE

    def get_target_language_code(self, target_language):
        return self.lang_code[target_language.lower().capitalize()]

    def translate(self, text, target_language, source_language="English"):
        """
        언어 코드가 아니라 국가 이름을 적어야함
        https://cloud.google.com/python/docs/reference/translate/latest/summary_method
        """
        text = text.split("\n")
        try:
            target_language_code = self.get_target_language_code(target_language)
            source_language_code = self.get_target_language_code(source_language)

            translate_client = translate.Client()
            result = translate_client.translate(
                text,
                target_language=target_language_code,
                source_language=source_language_code,
                format_="text",
            )

            logger.info(
                f"\n================================================\n {__name__}"
            )
            print(result, flush=True)
            nomalized = "\n".join([elem["translatedText"] for elem in result])
            logger.info("\n================================================\n")
            logger.info(f"nomalized: {nomalized}")
            logger.info("\n================================================\n")
            return nomalized
        except:
            print("Failed to translate text using Google API.")
            return ""


class PapagoTranslater:
    def __init__(self, opt):
        self.opt = opt
        self.__client_id = self.opt.TextTranslator_papago_client_id
        self.__client_secret = self.opt.TextTranslator_papago_client_secret
        # self.__url = "https://openapi.naver.com/v1/papago/n2mt"
        # self.__url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        self.__url = self.opt.TextTranslator_papago_url
        self.lang_dict = PAPAGO_LANGUAGES

    def get_target_language_code(self, target_language):
        return self.lang_dict[target_language.lower().capitalize()]

    def translate(self, text, target_language, source_language="Korean"):
        # Build form data
        print(f"debugg 1")
        try:
            form = {
                "source": self.get_target_language_code(source_language),
                "target": self.get_target_language_code(target_language),
                "text": text,
            }
            data = urllib.parse.urlencode(form).encode("utf-8")

            # 1) Try NCP API Gateway endpoint
            request = urllib.request.Request(self.__url, data=data)
            request.add_header(
                "Content-Type", "application/x-www-form-urlencoded; charset=UTF-8"
            )
            request.add_header("X-NCP-APIGW-API-KEY-ID", self.__client_id)
            request.add_header("X-NCP-APIGW-API-KEY", self.__client_secret)

            try:
                response = urllib.request.urlopen(request)
                if response.getcode() == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    translated_text = (
                        payload.get("message", {})
                        .get("result", {})
                        .get("translatedText", "")
                    )
                    return translated_text
            except Exception as e:
                logger.error(f"error: {e}")
                # Fall through to legacy endpoint
                pass
            # 2) Fallback: legacy Papago endpoint
            legacy_url = "https://openapi.naver.com/v1/papago/n2mt"
            legacy_req = urllib.request.Request(legacy_url, data=data)
            legacy_req.add_header(
                "Content-Type", "application/x-www-form-urlencoded; charset=UTF-8"
            )
            legacy_req.add_header("X-Naver-Client-Id", self.__client_id)
            legacy_req.add_header("X-Naver-Client-Secret", self.__client_secret)

            try:
                legacy_resp = urllib.request.urlopen(legacy_req)

                if legacy_resp.getcode() == 200:
                    payload = json.loads(legacy_resp.read().decode("utf-8"))
                    translated_text = (
                        payload.get("message", {})
                        .get("result", {})
                        .get("translatedText", "")
                    )

                    return translated_text
            except Exception as e:
                logger.error(f"error: {e}")
            return ""
        except Exception:
            logger.error(f"error: {e}")
            return ""


class DeepLTranslater:
    def __init__(self, opt):
        self.opt = opt
        self.__auth_key = self.opt.TextTranslator_deepl_auth_key
        self.__translator = deepl.Translator(self.__auth_key)
        self.lang_code = DEEPL_LANGUAGES

    def get_target_language_code(self, target_language):
        return self.lang_code[target_language.lower().capitalize()]

    # can be same as target_language, but can be additionally specified for
    # English (EN-GB / EN-US) and Portuguese (PT-BR / PT-PT)
    def get_source_language_code(self, source_language):
        return self.lang_code[source_language.lower().capitalize()]

    def translate(self, text, target_language, source_language="Korean"):
        source = self.get_source_language_code(source_language)
        target = self.get_target_language_code(target_language)
        # If multiple lines, call DeepL with list input to preserve segments.
        # logger.info(f"text: {text}")
        if "\n" in text:
            segments = text.split("\n")
            results = self.__translator.translate_text(
                segments,
                source_lang=source,
                target_lang=target,
                split_sentences="nonewlines",
                preserve_formatting=True,
            )
            logger.info(f"results: {results}")
            value_for_logger = "\n".join([r.text for r in results])
            logger.info(f"results.s results.text: {value_for_logger}")
            # results is a list of TextResult
            return "\n".join([r.text for r in results])
        else:
            translation = self.__translator.translate_text(
                text,
                source_lang=source,
                target_lang=target,
                split_sentences="nonewlines",
                preserve_formatting=True,
            )
            logger.info(f"translation: {translation}")
            logger.info(f"translation.text: {translation.text}")
            return translation.text


class BaiduTranslater:
    def __init__(self, opt):
        self.opt = opt


class OpenAITranslater:

    class BatchTranslationResponseFormatter(BaseModel):
        translations: List[str]
        source_language: str
        target_language: str

    def __init__(
        self,
        opt,
        response_formatter_class=BatchTranslationResponseFormatter,
    ):
        self.opt = opt
        self.client = OpenAI(api_key=self.opt.TextTranslator_openai_api_key)
        self.response_formatter_class = response_formatter_class

    def get_target_language_code(self, target_language):
        return target_language.lower().capitalize()

    def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = "ko",
        prompt: str = "",
    ) -> str:
        """
        texts: 번역할 문자열 배열
        prompt: 추가 번역 지시문(예: '공손하고 자연스럽게')
        target: 목표 언어 (예: 'en', 'ja', 'ko')
        source: 원본 언어(미지정 시 자동 감지)
        """

        print(f"text in openai translator: {text} \n\n")
        print(f"target_language in openai translator: {target_language} \n\n")
        print(f"source_language in openai translator: {source_language} \n\n")

        text = text.split("\n")

        # 공식 문서와 동일한 방식: responses.parse + text_format=BaseModel
        response = self.client.responses.parse(
            model="gpt-5-mini",
            reasoning={"effort": "low"},
            instructions=(
                "You are a professional webtoon/manga translation specialist. "
                "Return ONLY JSON with field `translations: string[]`. "
                "Preserve tags/placeholders and line breaks. "
                ""
                "WEBTOON TRANSLATION GUIDELINES: "
                "1. SOUND EFFECTS (의성어): Translate onomatopoeia naturally to target language equivalents "
                "   (e.g., '쿵' → 'THUD', '쨍그랑' → 'CLANG', '푸슉' → 'WHOOSH') "
                "2. EXPRESSIVE SOUNDS (의태어): Capture emotional/action sounds appropriately "
                "   (e.g., '훌쩍훌쩍' → 'sniff sniff', '두근두근' → 'thump thump') "
                "3. GENRE & TONE INFERENCE: Analyze context to determine genre (romance, action, comedy, horror) "
                "   and character personality (formal/casual, aggressive/gentle, modern/classical) "
                "4. CHARACTER VOICE: Maintain consistent speech patterns and personality traits "
                "5. CULTURAL ADAPTATION: Adapt cultural references while preserving original meaning "
                "6. DIALOGUE NATURALNESS: Make conversations sound natural in target language "
                ""
                "IMPORTANT: Translate ALL content including proper nouns naturally. "
                "Do not keep source language words - make everything sound natural in the target language. "
                f"Additional style guidance: {prompt}"
            ),
            input=[
                {
                    "role": "user",
                    "content": (
                        f"Translate the following array to {target_language}. "
                        f"Source language: {source_language or 'auto'}. "
                        "Make sure output length and order exactly match input.\n\n"
                        f"INPUT:\n{text}"
                    ),
                },
            ],
            # ⬇️ 여기서 모델이 만든 JSON을 Pydantic으로 자동 검증/파싱
            text_format=self.response_formatter_class,
        )
        logger.info(f"response from openai: {response.output_parsed} \n\n")
        # 다른 번역기와 동일한 형태로 맞추기 위해 문자열로 조인
        return "\n".join(response.output_parsed.translations)


class TextTranslator:
    def __init__(self, opt):
        self.opt = opt
        self.google = self._get_google_translate()
        self.papago = self._get_papago_translate()
        self.deepl = self._get_deepl_translate()
        self.openai = self._get_openai_translate()

    def _get_google_translate(self):
        try:
            googleTranslate = GoogleTranslater(self.opt)
            return googleTranslate
        except:
            print("Cannot initialize Google Translate Service. ")
            return None

    def _get_papago_translate(self):
        try:
            papagoTranslate = PapagoTranslater(self.opt)
            return papagoTranslate
        except:
            print("Cannot initialize Papago Translate Service. ")
            return None

    # NOT SUPPORTED IN KOREA!
    def _get_deepl_translate(self):
        try:
            deeplTranslate = DeepLTranslater(self.opt)
            return deeplTranslate
        except:
            # print("Cannot initialize DeepL Translate Service. ")
            return None

    def _get_openai_translate(self):
        try:
            openaiTranslate = OpenAITranslater(self.opt)
            return openaiTranslate
        except:
            print("Cannot initialize OpenAI Translate Service. ")
            return None

    def _remove_empty_lines_from_start_and_end(self, list_of_strings):
        start_empty_strings_count = 0
        end_empty_strings_count = 0
        for s in list_of_strings:
            if len(s) == 0:
                start_empty_strings_count += 1
            else:
                break
        for s in reversed(list_of_strings):
            if len(s) == 0:
                end_empty_strings_count += 1
            else:
                break
        if end_empty_strings_count != 0:
            return (
                list_of_strings[start_empty_strings_count:-end_empty_strings_count],
                start_empty_strings_count,
                end_empty_strings_count,
            )
        return (
            list_of_strings[start_empty_strings_count:],
            start_empty_strings_count,
            end_empty_strings_count,
        )

    def _pad_string_with_empty_lines(
        self, list_of_strings, start_empty_strings_count, end_empty_strings_count
    ):
        for _ in range(start_empty_strings_count):
            list_of_strings.insert(0, "")
        for _ in range(end_empty_strings_count):
            list_of_strings.append("")
        return list_of_strings

    def translate(
        self, list_of_strings, target_language="English", source_language="Korean"
    ):
        list_of_strings, list_of_strings_shape = array2d_to_array1d(list_of_strings)
        list_of_strings, start_empty_strings_count, end_empty_strings_count = (
            self._remove_empty_lines_from_start_and_end(list_of_strings)
        )
        translation_one_string = "\n".join(list_of_strings)

        try:
            translation_google = self.google.translate(
                text=translation_one_string,
                target_language=target_language,
                source_language=source_language,
            )
            output_translation_google = str(translation_google).split("\n")
            output_translation_google = self._pad_string_with_empty_lines(
                output_translation_google,
                start_empty_strings_count,
                end_empty_strings_count,
            )
            output_translation_google = array1d_to_array2d(
                output_translation_google, list_of_strings_shape
            )
        except:
            dummy_translation = [
                ""
                for _ in range(
                    len(list_of_strings)
                    + start_empty_strings_count
                    + end_empty_strings_count
                )
            ]
            dummy_translation = array1d_to_array2d(
                dummy_translation, list_of_strings_shape
            )
            output_translation_google = dummy_translation

        try:
            translation_papago = self.papago.translate(
                text=translation_one_string,
                target_language=target_language,
                source_language=source_language,
            )
            output_translation_papago = str(translation_papago).split("\n")
            output_translation_papago = self._pad_string_with_empty_lines(
                output_translation_papago,
                start_empty_strings_count,
                end_empty_strings_count,
            )
            output_translation_papago = array1d_to_array2d(
                output_translation_papago, list_of_strings_shape
            )
        except:
            dummy_translation = [
                ""
                for _ in range(
                    len(list_of_strings)
                    + start_empty_strings_count
                    + end_empty_strings_count
                )
            ]
            dummy_translation = array1d_to_array2d(
                dummy_translation, list_of_strings_shape
            )
            output_translation_papago = dummy_translation

        try:
            translation_deepl = self.deepl.translate(
                text=translation_one_string,
                target_language=target_language,
                source_language=source_language,
            )
            output_translation_deepl = str(translation_deepl).split("\n")
            output_translation_deepl = self._pad_string_with_empty_lines(
                output_translation_deepl,
                start_empty_strings_count,
                end_empty_strings_count,
            )
            output_translation_deepl = array1d_to_array2d(
                output_translation_deepl, list_of_strings_shape
            )
        except:
            dummy_translation = [
                ""
                for _ in range(
                    len(list_of_strings)
                    + start_empty_strings_count
                    + end_empty_strings_count
                )
            ]
            dummy_translation = array1d_to_array2d(
                dummy_translation, list_of_strings_shape
            )
            output_translation_deepl = dummy_translation

        output_translation = {
            "google": output_translation_google,
            "papago": output_translation_papago,
            "deepl": output_translation_deepl,
        }
        return output_translation
