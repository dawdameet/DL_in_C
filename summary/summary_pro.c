#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>

#define MAX_SENTENCES 1000
#define MAX_WORDS 10000
#define MAX_STOPWORDS 500
#define WORDLEN 64
#define MAX_TITLE_WORDS 100

typedef struct {
    char text[1024];
    double score;
    int index;
} Sentence;

typedef struct {
    char word[WORDLEN];
    double tf;
    double idf;
    double tfidf;
} WordStat;

char stopwords[MAX_STOPWORDS][WORDLEN];
int stopword_count = 0;

// ========================== Utility ==========================
void to_lowercase(char *s) { for (; *s; s++) *s = tolower(*s); }

int is_stopword(const char *word) {
    for (int i = 0; i < stopword_count; i++)
        if (strcmp(stopwords[i], word) == 0) return 1;
    return 0;
}

void load_stopwords(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return;
    while (fscanf(f, "%63s", stopwords[stopword_count]) != EOF)
        stopword_count++;
    fclose(f);
}

void strip_suffix(char *word) {
    int len = strlen(word);
    if (len > 4 && strcmp(word + len - 3, "ing") == 0) word[len - 3] = '\0';
    else if (len > 3 && strcmp(word + len - 2, "ed") == 0) word[len - 2] = '\0';
    else if (len > 3 && strcmp(word + len - 1, "s") == 0) word[len - 1] = '\0';
}

// ========================== Tokenization ==========================
int tokenize_sentence(char *sentence, char tokens[MAX_WORDS][WORDLEN]) {
    int count = 0;
    char *p = strtok(sentence, " ,.;:!?()\n\r\t");
    while (p && count < MAX_WORDS) {
        to_lowercase(p);
        strip_suffix(p);
        if (!is_stopword(p) && strlen(p) > 1)
            strncpy(tokens[count++], p, WORDLEN);
        p = strtok(NULL, " ,.;:!?()\n\r\t");
    }
    return count;
}

int split_sentences(const char *text, Sentence sentences[MAX_SENTENCES]) {
    int count = 0;
    const char *start = text;
    for (const char *p = text; *p; p++) {
        if (*p == '.' || *p == '!' || *p == '?') {
            int len = p - start + 1;
            if (len > 5 && count < MAX_SENTENCES) {
                strncpy(sentences[count].text, start, len);
                sentences[count].text[len] = '\0';
                sentences[count].index = count;
                sentences[count].score = 0;
                count++;
            }
            start = p + 1;
        }
    }
    return count;
}

// ========================== TF-IDF ==========================
void compute_tfidf(Sentence *sentences, int n_sentences) {
    char all_words[MAX_WORDS][WORDLEN];
    int word_doc_count[MAX_WORDS] = {0};
    int total_unique = 0;

    // Build document frequency
    for (int i = 0; i < n_sentences; i++) {
        char temp[1024];
        strcpy(temp, sentences[i].text);
        char tokens[MAX_WORDS][WORDLEN];
        int n_tokens = tokenize_sentence(temp, tokens);
        int local_mark[MAX_WORDS] = {0};
        for (int j = 0; j < n_tokens; j++) {
            int found = -1;
            for (int k = 0; k < total_unique; k++)
                if (strcmp(tokens[j], all_words[k]) == 0) { found = k; break; }
            if (found == -1 && total_unique < MAX_WORDS) {
                strcpy(all_words[total_unique], tokens[j]);
                word_doc_count[total_unique++] = 1;
            } else if (!local_mark[found]) {
                word_doc_count[found]++;
                local_mark[found] = 1;
            }
        }
    }

    // Compute TF-IDF per sentence
    #pragma omp parallel for
    for (int i = 0; i < n_sentences; i++) {
        char temp[1024];
        strcpy(temp, sentences[i].text);
        char tokens[MAX_WORDS][WORDLEN];
        int n_tokens = tokenize_sentence(temp, tokens);

        double tfidf_sum = 0;
        for (int j = 0; j < n_tokens; j++) {
            for (int k = 0; k < total_unique; k++) {
                if (strcmp(tokens[j], all_words[k]) == 0) {
                    double idf = log((double)n_sentences / (1 + word_doc_count[k]));
                    tfidf_sum += idf;
                    break;
                }
            }
        }
        sentences[i].score = tfidf_sum;
    }
}

// ========================== Cosine Similarity ==========================
double cosine_similarity(Sentence *a, Sentence *b) {
    char tmp1[1024], tmp2[1024];
    strcpy(tmp1, a->text);
    strcpy(tmp2, b->text);
    char tokens1[MAX_WORDS][WORDLEN], tokens2[MAX_WORDS][WORDLEN];
    int n1 = tokenize_sentence(tmp1, tokens1);
    int n2 = tokenize_sentence(tmp2, tokens2);
    int overlap = 0;
    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n2; j++)
            if (strcmp(tokens1[i], tokens2[j]) == 0) overlap++;
    if (n1 == 0 || n2 == 0) return 0;
    return (double)overlap / sqrt((double)n1 * n2);
}

// ========================== Title Boost ==========================
void apply_title_boost(Sentence *sentences, int n_sentences, const char *title) {
    char tmp[512];
    strcpy(tmp, title);
    char title_tokens[MAX_TITLE_WORDS][WORDLEN];
    int n_title = tokenize_sentence(tmp, title_tokens);

    for (int i = 0; i < n_sentences; i++) {
        for (int j = 0; j < n_title; j++) {
            if (strstr(sentences[i].text, title_tokens[j])) {
                sentences[i].score *= 1.1;
                break;
            }
        }
    }
}

// ========================== Position Bias ==========================
void apply_position_bias(Sentence *sentences, int n_sentences) {
    for (int i = 0; i < n_sentences; i++) {
        double pos_factor = 1.0 / (1.0 + sentences[i].index);
        sentences[i].score = 0.8 * sentences[i].score + 0.2 * pos_factor;
    }
}

// ========================== Comparator ==========================
int cmp_score(const void *a, const void *b) {
    double diff = ((Sentence*)b)->score - ((Sentence*)a)->score;
    return (diff > 0) - (diff < 0);
}
int cmp_index(const void *a, const void *b) {
    return ((Sentence*)a)->index - ((Sentence*)b)->index;
}

// ========================== Main ==========================
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file> [-n N] [-stop stopwords.txt] [-title \"Title\"]\n", argv[0]);
        return 1;
    }

    int n_summary = 5;
    char stopfile[128] = "";
    char title[256] = "";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_summary = atoi(argv[++i]);
        else if (strcmp(argv[i], "-stop") == 0 && i + 1 < argc) strcpy(stopfile, argv[++i]);
        else if (strcmp(argv[i], "-title") == 0 && i + 1 < argc) strcpy(title, argv[++i]);
    }

    if (strlen(stopfile)) load_stopwords(stopfile);

    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("File open"); return 1; }

    char text[65536];
    fread(text, 1, sizeof(text), f);
    fclose(f);

    Sentence sentences[MAX_SENTENCES];
    int n_sentences = split_sentences(text, sentences);

    compute_tfidf(sentences, n_sentences);
    apply_position_bias(sentences, n_sentences);
    if (strlen(title)) apply_title_boost(sentences, n_sentences, title);

    qsort(sentences, n_sentences, sizeof(Sentence), cmp_score);

    // Redundancy filtering
    Sentence summary[MAX_SENTENCES];
    int count = 0;
    for (int i = 0; i < n_sentences && count < n_summary; i++) {
        int redundant = 0;
        for (int j = 0; j < count; j++) {
            if (cosine_similarity(&sentences[i], &summary[j]) > 0.8) {
                redundant = 1; break;
            }
        }
        if (!redundant) summary[count++] = sentences[i];
    }

    qsort(summary, count, sizeof(Sentence), cmp_index);

    printf("===== SUMMARY =====\n");
    for (int i = 0; i < count; i++)
        printf("- %s\n", summary[i].text);

    return 0;
}
