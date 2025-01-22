import nltk
from nltk import word_tokenize

# tokenise a text using nltk
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


# define a function to get the edit distance between two words
def get_edit_distance(word1, word2):
    return nltk.edit_distance(word1, word2)

def chunk_text(text, max_tokens=100):
    sentences = nltk.sent_tokenize(text)
    #print(len(sentences))

    # print("chunking the text....")
    # Group sentences into small chunk of text whose token length should not be over max token length.
    # Split the input text into chunks of max_chunk_length
    chunks = []
    current_chunk = []
    # Chunk the sentences based on the maximum token length
    for sentence in sentences:
        sentence_tokens = nltk.tokenize.word_tokenize(sentence)
        if len(current_chunk) + len(sentence_tokens) < max_tokens :
            current_chunk.extend(sentence_tokens)
        else:
            chunks.append(current_chunk)
            current_chunk = sentence_tokens

    if current_chunk:
        chunks.append(current_chunk)

    # Convert token IDs back to text
    chunked_texts = [' '.join(chunk) for chunk in chunks]
    # print(f"text is chunked into {len(grouped_sentences)} pieces")
    return chunked_texts


if __name__ == '__main__':
    test_text = "__RO BBERY AT A BARONET'S. v EDWARn PRING, twenty-seven, caipentbr, was bro1ught up on remand at the Greenwich Police-court, charged with stealing jewellery to the Value of �100, the pro- pertyofSirRobert Cunliffe, Bart.,2,L1P., of37 Lot ndes- street, Beigravia. Chief Inspector Phillips said there were a number of cbarges against the prisoner, all the robberies alleged being tinder similar circumstances. The prisoner was taken into custody, it appeared, on a charge of stealing a wxatch, and other articles, value �12, from the residence of Miss ,1Kene, 8 Belgrave- terrace, Lee. ThP prisoner called at the house and represented that lie had been sent by the landlord to see to the blinds. He *was given access to the bedrooms, and after he had gone a watch, two rings, - and two lockets were eiliseed. His statement of having been sent to the house was found to be false. The prisoner was ultimately arrested on this charge, and he was sub- sequently charged withltherobbery fromiSir R, Cunliffe's upon information given by a young woman charged before the magistrate with the unlawvful possession of a diamond ring. Robert Yfoellam, footman to Sir Robert, deposed thbt on March 28 the prisoner came to thle house and said ha had rome to measure the blinds. Witn6ss said new blinds were not rdqoiifed, ,whereupon prkoner said he had been sent to put soime new springs. He was permitted to go upstairs, and witness afterwards let him out at the Front door. The house was being painted. Sir Robert Cunliffe was sworn, and said a quiantity ofjewellery *vhs missed on theevening of March 23 by Lady Cunliffe, with whom he had been out during the day. It was taken from the drawets in her &assin-table. He had since identified a pair of paste buckles, gold locket, emerald ring, two diamold rings pair of crystal studs, Nor- mandy cross, single taste bucklel, earbuncle locket, pair of earrings, and two eter rings. Other articles of jwllery and a purse containing �4 in gold and a �5 note were stoleh. Jane Pring, mother of the pri- sonller proved pledging several of the articles at hWr son's request. The prisoner was further remanded for a week."
    texts = chunk_text(test_text, max_tokens=60)
    print(texts[0])
