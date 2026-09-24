import re
import statistics

#splits the sentences by sentence ending punctuation, record sentence length

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
    "e.g", "i.e", "cf", "vs", "etc", "al", "fig", "eq", "vol",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

_WORD = re.compile(r"[A-Za-z']+")

def _ends_inabbreviation(buffer: str) -> bool:
    stripped = buffer.rstrip('"\')]')
    words = stripped.split()
    if not words:
        return False
    last = words[-1].rstrip('.').lower()

    if last in ABBREVIATIONS:
        return True
    if len(last) == 1 and last.isalpha():
        return True
    if re.fullmatch(r'\d+\.', stripped):
        return True
    return False


#function called split_sentence - intakes text as string - outputs a list of strings
#that list of strings is the list of sentences
def split_sentences(text: str) ->list[str]:
    #I am going to split by 'oversplitting' then repairing the broken sentences
    text = " ".join(text.split())
    if not text:
        return []

    sentences = []
    buffer = ""

    for piece in _BOUNDARY.split(text):
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if _ends_inabbreviation(buffer):
            continue
        sentences.append(buffer)
        buffer = ""

    if buffer:
        sentences.append(buffer)

    return sentences

def count_words(sentence: str) -> int:
    return len(_WORD.findall(sentence))



def burstiness(text: str) -> float | None:
    lengths = [count_words(s) for s in split_sentences(text)]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 5:
        return None
    return statistics.pstdev(lengths) / statistics.fmean(lengths)


## Later features Grammar checker
# Track the use of punctuations in order to track the use of grammar in the text.
def punctuation_usage(text: str) -> dict[str, int]:
    punctuation_counts = {}
    for char in text:
        if char in '.,;:!?':
            punctuation_counts[char] = punctuation_counts.get(char, 0) + 1
    return punctuation_counts

def proper_punctuation_usage(text: str) -> dict[str, int]:
    sentences = split_sentences(text)
    proper_counts = {}
    for sentence in sentences:
        if sentence and sentence[0].isupper() and sentence[-1] in '.!?':
            proper_counts['proper'] = proper_counts.get('proper', 0) + 1
        else:
            proper_counts['improper'] = proper_counts.get('improper', 0) + 1
    return proper_counts



## Vocabulary Richness Feature
# Measures the diversity of words used in the text.

def vocabulary_richness(text: str) -> float | None:
    words = _WORD.findall(text.lower())
    if not words:
        return None
    unique_words = set(words)
    return len(unique_words) / len(words)

# Second additional function to track words that the author uses frequently,
# this can help to determine wether the author is working with a limited human
# vocab or an infinite library of words often related to AI usage.





FEATURES = {
    "burstiness": burstiness,                                  ## Measures the variability in sentence lengths
    "proper_punctuation_usage": proper_punctuation_usage,      ## Measures the grammar usage in the sentences

}



