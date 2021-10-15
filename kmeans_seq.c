#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K 2 // Number of Clusters
#define ITERATIONS 20 // Number of iterations of KMeans
#define NUM_ELEMENTS 2000 // Number of datapoints
#define D 2 // Dimension of data

double calculate_distance(double *d1, double *d2){
    double dist = 0;
	for (int i = 0; i < D; i++)
		dist += (d1[i] - d2[i])*(d1[i] - d2[i]);
	return dist;
}

double* element_k(double *data, int col){
	double* col_k = (double *)malloc(D*sizeof(double));
	for(int i=0; i<D; i++){
		col_k[i] = data[col*D + i];
	}
	return col_k;
}

int calculate_closest_centroid(double *data, double *means){
	double min_dist = RAND_MAX, dist_i;
	int closest_centroid;
	for(int i=0; i<K; i++){
		dist_i = calculate_distance(data, element_k(means,i));
		if(dist_i<min_dist){
			min_dist = dist_i;
			closest_centroid = i;
		}
	}
	return closest_centroid;
}

double* compute_centroids(double *data, int* cluster_label){
	double *means = (double *)malloc(K*D*sizeof(double));
	int *cluster_count = (int *)malloc(K*sizeof(int));
	for(int i=0; i<NUM_ELEMENTS; i++){
		cluster_count[cluster_label[i]]++;
		for(int j=0; j<D; j++){
			means[cluster_label[i]*D +j] += data[i*D +j];
		}		
	}
	for(int i=0; i<K; i++){
		for(int j=0; j<D; j++){
			means[i*D +j] /= cluster_count[i];
		}
	}
	return means;
}

int calc_num_moved_pts(int *label_prev, int *label_ct){
	int count = 0;

	for(int i=0; i<NUM_ELEMENTS; i++){
		if(label_prev[i] != label_ct[i]){
			count++;
		}
	}
	return count;
}

int main(){
	double x1, x2;
	double *x = (double *)malloc(NUM_ELEMENTS*D*sizeof(double));
	char line[256];
	FILE *fptr;
	fptr = fopen("file_2k.txt","r");
	if(fptr == NULL){
    	printf("File cannot be opened \n");
		exit(0);
	}
	int i=0;
	while(!feof(fptr)){
		fgets(line,sizeof(line),fptr);
		sscanf(line,"%lf, %lf",&x1,&x2);
		x[i] = x1;
		x[i+1] = x2;
		i = i+D;
	}

	// K Means Clustering
	// Initialize means around K random datapoints
	double *means = (double *)malloc(K*D*sizeof(double));
	int *cluster_label_ct = (int *)malloc(NUM_ELEMENTS*sizeof(int));
	int *cluster_label_prev = (int *)malloc(NUM_ELEMENTS*sizeof(int));
	int moved_pts;

	for(int i=0;i<K;i++){
		int rand_num = rand()%NUM_ELEMENTS;
		for(int j=0;j<D;j++){
			float rand_float = ((float)rand()/(float)(RAND_MAX)) * 5;
			means[i*D+j] = x[rand_num*D+j] + rand_float;
		}
	}

	printf("\nInitial Means \n");
	for(int i=0;i<K*D;i++){
		printf("%lf ",means[i]);
	}

	for(int t=0; t<ITERATIONS; t++){
		// Assign datapoints to clusters
		for(int i=0; i<NUM_ELEMENTS; i++){
			//assign pt to the closest centroid
			cluster_label_ct[i] = calculate_closest_centroid(element_k(x,i), means);
		}
		
		// Recompute the means
		double *new_means = (double *)malloc(K*D*sizeof(double));
		new_means = compute_centroids(x, cluster_label_ct);
		// printf("\n Means : Iteration : %d \n",t);
		// for(int i=0;i<K*D;i++){
		// 	printf("%lf ",new_means[i]);
		// }
		means = new_means;
		if(t>0){
			moved_pts = calc_num_moved_pts(cluster_label_prev, cluster_label_ct);
			// printf("\n Number of points moved : %d \n",moved_pts);
		}
		if(moved_pts==0)
			break;
		for(int i=0; i<NUM_ELEMENTS; i++){
			cluster_label_prev[i] = cluster_label_ct[i];
		}
	}

	// Initialize variance and Lambda
	double *initial_var = (double *)malloc(K*D*D*sizeof(double));
	double *initial_lambda = (double *)malloc(K*sizeof(double));
	double *temp_diff = (double *)malloc(D*sizeof(double));
	int *cluster_count = (int *)malloc(K*sizeof(int));
	double temp;
	for(int i=0; i<NUM_ELEMENTS; i++){
		cluster_count[cluster_label_ct[i]]++;
		for(int m=0; m<D; m++){
			temp_diff[m] = x[i*D +m] - means[cluster_label_ct[i]*D +m];
		}
		for(int j=0; j<D; j++){
			for(int jj=j; jj<D; jj++){
				temp = temp_diff[j]*temp_diff[jj];
				initial_var[cluster_label_ct[i]*D*D + j*D + jj] += temp;
				if(j!=jj)
					initial_var[cluster_label_ct[i]*D*D + jj*D + j] += temp;
			}
		}
	}
	for(int i=0; i<K; i++){
		initial_lambda[i] = (double)cluster_count[i]/NUM_ELEMENTS;
		printf("\n %lf", initial_lambda[i]);
		for(int j=0; j<D*D; j++){
			initial_var[i*D*D + j] /= cluster_count[i];
		}
	}

	printf("\n Means \n");
	for(int i=0;i<K*D;i++){
		printf("%lf ",means[i]);
	}
	printf("\n Initial Covariance Matrix \n");
	for(int i=0; i<K; i++){
		printf(" Cluster %d\n", i);
		for(int j=0; j<D; j++){
			for(int k=0; k<D; k++){
				printf("%lf\t",initial_var[i*D*D + j*D + k]);
			}
			printf("\n");
		}
	}
	return 0;
}