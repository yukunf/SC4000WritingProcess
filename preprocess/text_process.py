from tqdm import tqdm
import pandas as pd
import re
import numpy as np


class TextProcessor:
    PUNCTUATION_MAP = {
        "SPACE": " ",
        "COMMA": ",",
        "DOUBLE_QUOTE": '"',
        "PERIOD": ".",
        "PARENTHESES_OPEN": "(",
        "PARENTHESES_CLOSE": ")",
        "SQUARE_BRACKET_OPEN": "[",
        "SQUARE_BRACKET_CLOSE": "]",
        "CURLY_BRACKET_OPEN": "{",
        "CURLY_BRACKET_CLOSE": "}",
        "EXCLAMATION_MARK": "!",
        "QUESTION_MARK": "?",
    }

    def insert_text(self, text, s, pos):
        return "".join((text[:pos], s, text[pos:]))

    def remove_text(self, text, s, pos):
        return "".join((text[:pos], text[pos + len(s):]))

    def replace_text(self, text, s1, s2, pos):
        return "".join((text[:pos], s2, text[pos + len(s1):]))

    def move_text(self, text, s, pos1, pos2):
        text = self.remove_text(text, s, pos1)
        text = self.insert_text(text, s, pos2)
        return text

    def split_to_word(self, s):
        s = s.lower()
        char_sep = "@"
        punctuation_chars = list(self.PUNCTUATION_MAP.values())
        for pun in punctuation_chars:
            s = s.replace(pun, char_sep)
        s_arr = re.split(char_sep, s)
        s_arr = [w for w in s_arr if "q" in w]
        return s_arr

    def split_to_sentence(self, s):
        s = s.lower()
        char_sep = "@"
        punctuation = [".", "!", "?"]
        for punc in punctuation:
            s = s.replace(punc, char_sep)
        s_arr = re.split(char_sep, s)
        s_arr = [w for w in s_arr if ("q" in w)]
        return s_arr

    def split_to_paragraph(self, s):
        s = s.lower()
        char_sep = "@"
        s_arr = re.split(char_sep, s)
        s_arr = [w for w in s_arr if ("q" in w)]
        return s_arr

    def change_punctuation(self, text):
        reverse_map = {v: k.lower()
                       for k, v in self.PUNCTUATION_MAP.items()}
        result = []
        for char in text:
            if char in reverse_map:
                result.append(' ' + reverse_map[char] + ' ')
            else:
                result.append(char)
        output = "".join(result)
        output = re.sub(r"\s+", " ", output).strip()

        return output


class EssayConstructor:
    def __init__(self):
        self.text_processor = TextProcessor()

    def recon_writing(self, df):
        res_all = []
        len_texts = []
        sentence_counts = []
        paragraph_counts = []

        res = ""
        prev_idx = ""

        temp_df = df[['id', 'activity', 'up_event', 'text_change',
                      'cursor_position', 'word_count']].values

        for row in tqdm(temp_df):
            idx = str(row[0])
            activity, up_event, text_change = str(
                row[1]), str(row[2]), str(row[3])
            cursor_position, _ = int(row[4]), int(row[5])

            # new idx
            if idx != prev_idx:
                if prev_idx != "":
                    res_all.append(res)
                    len_texts.append(len_text)
                    sentence_counts.append(sentence_count)
                    paragraph_counts.append(paragraph_count)

                res, len_text, sentence_count, paragraph_count = "", 0, 0, 0
                prev_idx = idx

            if activity != "Nonproduction":
                # replace the newline character to n
                text_change = text_change.replace("@", "/").replace("\n", "n")

                if (activity == "Input") | (activity == "Paste"):
                    res = self.text_processor.insert_text(
                        res, text_change, cursor_position - len(text_change)
                    )

                elif activity == "Remove/Cut":
                    res = self.text_processor.remove_text(
                        res, text_change, cursor_position
                    )

                elif activity == "Replace":
                    before, after = text_change.split(" => ")
                    res = self.text_processor.replace_text(
                        res, before, after, cursor_position - len(after)
                    )

                elif "Move" in activity:
                    pos = [int(s) for s in re.findall(r"\d+", activity)]
                    # pos 0 start pos1 end pos2 start pos3 end
                    res = self.text_processor.move_text(
                        res, text_change, pos[0], pos[2]
                    )

                len_text = len(res)
                sentence_count = len(
                    self.text_processor.split_to_sentence(res))
                paragraph_count = len(
                    self.text_processor.split_to_paragraph(res))

            prev_up_event = up_event

        # append last essay data
        res_all.append(res)
        len_texts.append(len_text)
        sentence_counts.append(sentence_count)
        paragraph_counts.append(paragraph_count)

        return res_all, len_texts, sentence_counts, paragraph_counts


if __name__ == "__main__":
    df = pd.read_csv("data/train_logs_clean.csv")
    essay_constructor = EssayConstructor()
    reconstructed_texts, len_texts, sentence_counts, paragraph_counts = essay_constructor.recon_writing(
        df)
    idx = df["id"].unique()
    result_df = pd.DataFrame({"id": idx, "text": reconstructed_texts, "len_text": len_texts,
                             "sentence_count": sentence_counts, "paragraph_count": paragraph_counts})
    result_df.to_csv("data/train_logs_extracted_text.csv", index=False)
