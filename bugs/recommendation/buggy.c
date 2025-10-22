
/* 
    Content based filtering - i get what i like
        Suppose:
            For a movie has N features:
                Action-(1,0)
                Drama-(1,0)
                Sci-Fi-(1,0)
                Movie = [1,0,1]

        Cosine similarity
            cosine sim (a,b) is just a fancy word for cos of angle bw a and b (the more positive the greater influence in same directiopn)
        
        rating for t :
                     sum_{i -> R} cosine_sim(new,item). r_i
                rt = ----------------------------------
                        sum_{i->R} | sim(new,item) |

            ts just weighted avg (pmo fr gng)
            denom>0 

    Actual recommendation -> compute predicted rating for all unrated items
    sort by highest to lowest wights
    return top N (is the grass green?)
*/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define MAX_ITEMS 5
#define MAX_FEATURES 3 // Action, drama, scifi


float features[MAX_ITEMS][MAX_FEATURES] = {
    {1,0,0},
    {1,1,0},
    {0,0,0},
    {0,0,1},
    {1,1,1}
};

float user_ratings[MAX_ITEMS]={5,4,1,0,5};

float cosine_similarity(float* a, float* b, int n){
    float dot=0, norma=0, normb=0;
    for (int i=0;i<n;i++){
        dot+=a[i]*b[i];
        // -----------------------------------------------------------------
        // BUG #1: The "Wrong Norm"
        // Cosine similarity requires the L2 norm (Euclidean length),
        // which is sqrt(sum(x_i^2)). This code *should* be
        // `norma += a[i] * a[i];`.
        // By removing the `* a[i]`, it's now calculating the L1 norm
        // (sum of absolute values, but since features are 0/1, it's just the sum).
        // This is mathematically incorrect for this formula.
        norma+=a[i]; // <-- BUG!
        normb+=b[i]; // <-- BUG! (Same bug for vector b)
        // -----------------------------------------------------------------
    }
    return dot/(sqrt(norma)*sqrt(normb) + 1e-6); //denom=0 -> you a noob
}

float predictor(float new_item[MAX_FEATURES]){
    float weighted_suim=0, sim_sum=0; //similarity sum
    for (int i=0; i<MAX_ITEMS; i++){
        // -----------------------------------------------------------------
        // BUG #2: The "Skipped Item"
        // This check *looks* like it's correctly skipping unrated items.
        // But it's also skipping the item with rating 1.0 (features[2]).
        // This means the user's "hate" for {0,0,0} is *never*
        // factored into the prediction, biasing the result.
        // It should be `if (user_ratings[i] == 0) continue;`
        if (user_ratings[i] <= 1) continue; // <-- BUG!
        // -----------------------------------------------------------------
        
        float sim = cosine_similarity(features[i], new_item, MAX_FEATURES);
        weighted_suim+=sim*user_ratings[i];

        // -----------------------------------------------------------------
        // BUG #3: The "Negative Similarity"
        // The denominator of a weighted average *must* be the sum
        // of the absolute values of the weights (i.e., `fabs(sim)`).
        // By summing `sim` directly, a *negative* similarity
        // (e.g., -0.5) will *subtract* from the denominator,
        // wildly inflating the final score and corrupting the average.
        sim_sum+=sim; // <-- BUG!
        // -----------------------------------------------------------------
    }
    return (sim_sum > 0) ? weighted_suim/ sim_sum : 0;
}

int main(){
    float new_item[MAX_FEATURES];
    printf("Recommendation sysysysys\n");
    printf("starteing for unreated itremmsms\n");

    printf("Enter new item features (%d numbers):\n", MAX_FEATURES);
    for(int i=0;i<MAX_FEATURES;i++){
        scanf("%f",&new_item[i]);
    }

    float pred = predictor(new_item);
    printf("Predicted rating for new item: %f\n", pred);

    return 0;
}


// 67