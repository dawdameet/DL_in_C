#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TEXT 10000
#define MAX_SENTENCES 500
#define MAX_WORDS 10000
#define MAX_WORD_LEN 64

typedef struct {
    char word[MAX_WORD_LEN];
    int count;
} WordFreq;

typedef struct {
    char sentence[512];
    double score;
} SentenceScore;

WordFreq vocab[MAX_WORDS];
int vocab_size = 0;
int total_words = 0;

void to_lower(char *s) {
    for (int i = 0; s[i]; i++)
        s[i] = tolower((unsigned char)s[i]);
}

void clean_word(char *w) {
    int len = strlen(w);
    int j = 0;
    for (int i = 0; i < len; i++) {
        if (isalpha((unsigned char)w[i]))
            w[j++] = tolower((unsigned char)w[i]);
    }
    w[j] = '\0';
}

void add_word(char *word) {
    if (strlen(word) == 0) return;
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(vocab[i].word, word) == 0) {
            vocab[i].count++;
            total_words++;
            return;
        }
    }
    if (vocab_size < MAX_WORDS) {
        strcpy(vocab[vocab_size].word, word);
        vocab[vocab_size].count = 1;
        vocab_size++;
        total_words++;
    }
}

double get_word_prob(char *word) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(vocab[i].word, word) == 0)
            return (double)vocab[i].count / total_words;
    }
    return 1e-6; // unseen smoothing
}

int cmp_score(const void *a, const void *b) {
    SentenceScore *x = (SentenceScore*)a;
    SentenceScore *y = (SentenceScore*)b;
    if (y->score > x->score) return 1;
    if (y->score < x->score) return -1;
    return 0;
}

int main() {
    char text[MAX_TEXT];
    printf("Enter text:\n");
    fgets(text, sizeof(text), stdin);

    // Split into sentences
    SentenceScore sentences[MAX_SENTENCES];
    int sent_count = 0;

    char *sent = strtok(text, ".!?");
    while (sent && sent_count < MAX_SENTENCES) {
        strcpy(sentences[sent_count].sentence, sent);
        sentences[sent_count].score = 0;
        sent_count++;
        sent = strtok(NULL, ".!?");
    }

    // Build vocabulary
    for (int i = 0; i < sent_count; i++) {
        char s_copy[512];
        strcpy(s_copy, sentences[i].sentence);

        char *word = strtok(s_copy, " ,;\n");
        while (word) {
            clean_word(word);
            add_word(word);
            word = strtok(NULL, " ,;\n");
        }
    }

    // Score sentences (Naive Bayes style)
    for (int i = 0; i < sent_count; i++) {
        char s_copy[512];
        strcpy(s_copy, sentences[i].sentence);

        char *word = strtok(s_copy, " ,;\n");
        while (word) {
            clean_word(word);
            sentences[i].score += get_word_prob(word);
            word = strtok(NULL, " ,;\n");
        }
    }

    // Sort by score
    qsort(sentences, sent_count, sizeof(SentenceScore), cmp_score);

    int summary_count = sent_count < 3 ? sent_count : 3;
    printf("\n===== SUMMARY =====\n");
    for (int i = 0; i < summary_count; i++) {
        printf("- %s.\n", sentences[i].sentence);
    }

    return 0;
}
