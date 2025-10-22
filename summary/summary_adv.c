#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>

#define MAX_TEXT       20000
#define MAX_SENTENCES  1000
#define MAX_WORDS      20000
#define MAX_WORD_LEN   64

typedef struct {
    char word[MAX_WORD_LEN];
    int tf;  // term frequency
    int df;  // doc frequency (# sentences containing word)
} WordStat;

typedef struct {
    char sentence[512];
    double score;
    int original_index;
} SentenceScore;

WordStat vocab[MAX_WORDS];
int vocab_size = 0;
int sentence_count = 0;
SentenceScore sentences[MAX_SENTENCES];

// --- Dynamic stopword list ---
char **stopwords = NULL;
int stopword_count = 0;

// ---------- Helper Functions ----------
void load_stopwords(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Warning: Could not open stopword file '%s'. Using none.\n", filename);
        return;
    }

    char buf[64];
    while (fscanf(fp, "%63s", buf) == 1) {
        stopwords = realloc(stopwords, (stopword_count + 1) * sizeof(char*));
        stopwords[stopword_count] = strdup(buf);
        stopword_count++;
    }
    fclose(fp);
}

int is_stopword(const char *w) {
    for (int i = 0; i < stopword_count; i++)
        if (strcmp(w, stopwords[i]) == 0)
            return 1;
    return 0;
}

void free_stopwords() {
    for (int i = 0; i < stopword_count; i++) free(stopwords[i]);
    free(stopwords);
}

// Basic stemming: removes “ing”, “ed”, “es”, “s”
void stem_word(char *w) {
    int len = strlen(w);
    if (len > 4 && strcmp(&w[len - 3], "ing") == 0) w[len - 3] = '\0';
    else if (len > 3 && strcmp(&w[len - 2], "ed") == 0) w[len - 2] = '\0';
    else if (len > 3 && strcmp(&w[len - 2], "es") == 0) w[len - 2] = '\0';
    else if (len > 2 && w[len - 1] == 's') w[len - 1] = '\0';
}

void clean_word(char *w) {
    int j = 0;
    for (int i = 0; w[i]; i++)
        if (isalpha((unsigned char)w[i]))
            w[j++] = tolower((unsigned char)w[i]);
    w[j] = '\0';
    stem_word(w);
}

void trim(char *s) {
    int start = 0;
    while (isspace((unsigned char)s[start])) start++;
    if (start > 0) memmove(s, s + start, strlen(s + start) + 1);
    int end = strlen(s) - 1;
    while (end >= 0 && isspace((unsigned char)s[end])) s[end--] = '\0';
}

int find_or_add_word(const char *w) {
    for (int i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i].word, w) == 0) return i;
    if (vocab_size < MAX_WORDS) {
        strcpy(vocab[vocab_size].word, w);
        vocab[vocab_size].tf = vocab[vocab_size].df = 0;
        return vocab_size++;
    }
    return -1;
}

int cmp_score(const void *a, const void *b) {
    double s1 = ((SentenceScore*)a)->score;
    double s2 = ((SentenceScore*)b)->score;
    return (s2 > s1) - (s2 < s1); // descending
}

int cmp_original_order(const void *a, const void *b) {
    const SentenceScore *sa = (const SentenceScore*)a;
    const SentenceScore *sb = (const SentenceScore*)b;
    return (sa->original_index - sb->original_index);
}

// ---------- Main ----------
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.txt> [-n N] [-stop stopwords.txt]\n", argv[0]);
        return 1;
    }

    int top_n = 3;
    const char *stopfile = NULL;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            top_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-stop") == 0 && i + 1 < argc)
            stopfile = argv[++i];
    }

    if (stopfile) load_stopwords(stopfile);

    FILE *fp = fopen(argv[1], "r");
    if (!fp) { perror("File open failed"); return 1; }

    char text[MAX_TEXT] = {0};
    fread(text, 1, sizeof(text) - 1, fp);
    fclose(fp);

    // Split into sentences
    char *p = strtok(text, ".!?");
    while (p && sentence_count < MAX_SENTENCES) {
        trim(p);
        if (strlen(p) > 0) {
            strcpy(sentences[sentence_count].sentence, p);
            sentences[sentence_count].score = 0;
            sentences[sentence_count].original_index = sentence_count;
            sentence_count++;
        }
        p = strtok(NULL, ".!?");
    }

    // --- Build vocab (TF + DF) ---
    for (int s = 0; s < sentence_count; s++) {
        char seen_in_sent[MAX_WORDS] = {0};
        char copy[512]; strcpy(copy, sentences[s].sentence);
        char *tok = strtok(copy, " ,;\n\t");
        while (tok) {
            clean_word(tok);
            if (strlen(tok) && !is_stopword(tok)) {
                int idx = find_or_add_word(tok);
                if (idx >= 0) {
                    vocab[idx].tf++;
                    if (!seen_in_sent[idx]) {
                        vocab[idx].df++;
                        seen_in_sent[idx] = 1;
                    }
                }
            }
            tok = strtok(NULL, " ,;\n\t");
        }
    }

    // --- Parallel scoring (TF-IDF log-sum) ---
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < sentence_count; s++) {
        char copy[512]; strcpy(copy, sentences[s].sentence);
        char *tok = strtok(copy, " ,;\n\t");
        double local_score = 0;
        while (tok) {
            clean_word(tok);
            if (strlen(tok) && !is_stopword(tok)) {
                for (int i = 0; i < vocab_size; i++) {
                    if (strcmp(vocab[i].word, tok) == 0) {
                        double tf = vocab[i].tf;
                        double idf = log((double)sentence_count / (1 + vocab[i].df));
                        double weight = tf * idf;
                        local_score += log(weight + 1e-6);
                        break;
                    }
                }
            }
            tok = strtok(NULL, " ,;\n\t");
        }
        sentences[s].score = local_score;
    }

    // Sort by descending score
    qsort(sentences, sentence_count, sizeof(SentenceScore), cmp_score);

    int n = (sentence_count < top_n) ? sentence_count : top_n;
    SentenceScore *selected = malloc(sizeof(SentenceScore) * n);
    memcpy(selected, sentences, sizeof(SentenceScore) * n);

    // Restore original order for readability
    qsort(selected, n, sizeof(SentenceScore), cmp_original_order);

    printf("===== SUMMARY =====\n");
    for (int i = 0; i < n; i++)
        printf("- %s.\n", selected[i].sentence);

    free(selected);
    free_stopwords();
    return 0;
}
