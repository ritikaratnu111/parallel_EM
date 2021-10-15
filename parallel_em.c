#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define K 4 // Number of Clusters
#define ITERATIONS 20 // Number of iterations of KMeans
#define NUM_ELEMENTS 1000 // Number of datapoints
#define D 1 // Dimension of data
#define EM_ITERATIONS 10 

double matMul (double*x, double*v){
	double temp;
	temp = v[0]*pow(x[0],2) + 2*v[1]*x[0]*x[1] + v[3]*pow(x[1],2);
	printf("\nmatmul=%f",temp);
	return temp;  
}

double det (double * v){
	double temp;
	temp = v[0]*v[3] - pow(v[1],2);
	printf("\ndetm=%f",temp);
	return temp;
}

double phi(double *x, double *means, double *v){
	double val;
	if (D==1){
		val = exp(-pow(x[0]-means[0],2)/(2*v[0]))/pow(2*M_PI*v[0],0.5); 
	}
	else{
		for (int dim=0; dim<D; dim++){
			x[dim] = x[dim] - means[dim];
			}		
		val = exp(-pow(matMul(x,v),2))/pow(2*M_PI*det(v),0.5);
	}
	return val;
}
int main(int argc, char** argv){
	
    int MYRANK;
    int NO_OF_PROCS;
	int ARRAY_LENGTH;
	int READ_START;
	double x1, x2;
	double *x = (double *)malloc(NUM_ELEMENTS*D*sizeof(double));
	char line[256];
	//double send_array[3];
	//double recv_array[3];	
	FILE *fptr;
	fptr = fopen("P1M1L1.txt","r");
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
	
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
  	MPI_Comm_size(MPI_COMM_WORLD, &NO_OF_PROCS);
	double *means = (double *)malloc(K*D*sizeof(double));
	double *var = (double *)malloc(K*D*D*sizeof(double));
	double *lam = (double *)malloc(K*sizeof(double));

	// instead of clustering start
	means[0] = 3.781;
	means[1] = 7.957;
	means[2] = 0.014;
	means[3] = 12.121;

	lam[0] = 0.254 ;
	lam[1] = 0.27;
	lam[2] = 0.241;
	lam[3] = 0.235;

	var[0] = 1.187;
	var[1] = 1.397;
	var[2] = 1.378;
	var[3] = 1.645;

	// instead of clustering end

	if (MYRANK==0){	
		printf("\n Initialized Values:");
		for(int j=0; j<K; j++){
			printf("\n\n\n myrank=%d",MYRANK);
			printf("\nCLuster No = %d",j);
			printf("\nlambda= %f",lam[j]);	
			printf("\nmean= %f",means[j]);
			printf("\nvar=%f",var[j]);
			printf("\n");
		}
	}
	

	ARRAY_LENGTH = NUM_ELEMENTS/NO_OF_PROCS;
	READ_START = MYRANK * ARRAY_LENGTH;

	MPI_Barrier(MPI_COMM_WORLD);
	//printf("\n\n\n myrank=%d,array length=%d,read_start=%d",MYRANK,ARRAY_LENGTH,READ_START);
	
	double *gamma = (double *)malloc(ARRAY_LENGTH*K*sizeof(double));
	double *temp_x = (double *)malloc(D*sizeof(double));
	double *temp_mean = (double *)malloc(D*sizeof(double));
	double *temp_var = (double *)malloc(D*D*sizeof(double));

	for (int i=0; i<ARRAY_LENGTH; i++){
		double den = 0;
		for (int dim=0; dim<D; dim++){
			temp_x[dim] = x[READ_START + i*D + dim];
			}
				
		for(int j=0; j<K; j++){
			for (int dim=0; dim<D; dim++){
				temp_mean[dim] = means[j*D+dim];
				}
			
			for (int dim=0; dim<D*D; dim++){
				temp_var[dim] = var[j*D*D+dim]; 
				}

			gamma[i*K+j] = lam[j]*phi(temp_x,temp_mean,temp_var);
			//printf("\ngamma[%d*K+%d]=%f\n",i,j,gamma[i*K+j]);
			den = den + gamma[i*K+j];
			//printf("\nden = %f",den);
		}
				
		for(int j=0; j<K; j++){
			gamma[i*K+j] = gamma[i*K+j]/den;	
		}
				
	} // all initial values have been computed

	MPI_Barrier(MPI_COMM_WORLD);
	//printf("%d",MYRANK);

	//begin EM 
	for (int iter=0; iter<10; iter++){

		double send_array[3*K];
		double recv_array[3*K];
		
		for (int j=0; j<K; j++){
			double num_mean = 0;
			double num_var = 0;
			double den = 0;

			for (int i=0; i<ARRAY_LENGTH; i++){		
				for(int dim=0; dim<D; dim++){
					num_mean = num_mean + gamma[i*K+j]*x[READ_START+i*D+dim];        //num_mean
					num_var = num_var + gamma[i*K+j]*pow(x[READ_START+i*D+dim]-means[j*D+dim],2); //num_var
					den = den + gamma[i*K+j];                                            //den
				}				
			}
			MPI_Barrier(MPI_COMM_WORLD);
			send_array[j*3+0] = num_mean;
			send_array[j*3+1] = num_var;
			send_array[j*3+2] = den;

			MPI_Barrier(MPI_COMM_WORLD);
			
			MPI_Reduce( send_array +j*3 ,recv_array +j*3 , 3 , MPI_DOUBLE, MPI_SUM, 0 , MPI_COMM_WORLD );
				
			if (MYRANK == 0){
				num_mean = recv_array[j*3+0];
				num_var = recv_array[j*3+1];
				den = recv_array[j*3+2];
				lam[j] = den/NUM_ELEMENTS;    //den
				//printf("\nlambda of %d cluster:",j);
				//printf("%f",lam[j]);
				//printf("\nmeans of %d cluster:",j);
				for (int dim=0; dim<D; dim++){
					means[j*D+dim] = num_mean/den;
					//printf("%f",means[j*D+dim]);
				}		
			
				//printf("\nvar of %d cluster:",j);
				for (int dim=0; dim<D*D; dim++){
					var[j*D*D+dim] = num_var/den;
					//printf("%f",var[j*D*D+dim]);
				}
			}				
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(lam,K,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(means,D*K,MPI_DOUBLE,0,MPI_COMM_WORLD);		
		MPI_Bcast(var,D*D*K,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//MPI_Barrier(MPI_COMM_WORLD);
		for (int i=0; i<ARRAY_LENGTH; i++){
			double den = 0;
			for (int dim=0; dim<D; dim++)
				temp_x[dim] = x[READ_START + i*D+dim];
				
			for(int j=0; j<K; j++){		
				for (int dim=0; dim<D; dim++)
					temp_mean[dim] = means[j*D+dim];
				for (int dim=0; dim<D*D; dim++)
					temp_var[dim] = var[j*D*D+dim]; 
				
				gamma[i*K+j] = lam[j]*phi(temp_x,temp_mean,temp_var);
				den = den + gamma[i*K+j];
			}	
			for(int j=0; j<K; j++)
				gamma[i*K+j] = gamma[i*K+j]/den;		
		}
		MPI_Barrier(MPI_COMM_WORLD);

		} //end of iterations

		if (MYRANK==0){	
			for(int j=0; j<K; j++){
				printf("\n\n\n myrank=%d",MYRANK);
				printf("\nCLuster No = %d",j);
				printf("\nlambda= %f",lam[j]);	
				printf("\nmean= %f",means[j]);
				printf("\nvar=%f",var[j]);
				printf("\n");
			}
		}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}