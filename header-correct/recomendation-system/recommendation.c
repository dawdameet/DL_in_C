/* Content based filtering - i get what i like
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
        norma+=a[i]*a[i];
        normb+=b[i]*b[i];
    }
    return dot/(sqrt(norma)*sqrt(normb) + 1e-6); //denom=0 -> you a noob
}

float predictor(float new_item[MAX_FEATURES]){
    float weighted_suim=0, sim_sum=0; //similarity sum
    for (int i=0; i<MAX_ITEMS; i++){
        if (user_ratings[i] == 0) continue;
        float sim = cosine_similarity(features[i], new_item, MAX_FEATURES);
        weighted_suim+=sim*user_ratings[i];
        sim_sum+=fabs(sim);
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