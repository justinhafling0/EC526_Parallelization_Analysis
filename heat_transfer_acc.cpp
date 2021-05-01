// Based on the tutorial from mpitutorial.com/tutorials/mpi-hello-world/
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>

// Maximum number of iterations
#define ITER_MAX 1000000

// How often to check the relative residual
#define RESID_FREQ 1000

// The residual
#define RESID 1e-10

#define D_X 0.001
#define D_Y 0.001

#define L 256

#define CLOCKSPEED 2.4e9

// ALUMINUM       GOLD      SILVer     COPPER        DIAMOND
int PROPERTY_CP[]={903, 129, 235, 385, 509, 0};
int PROPERTY_K[] ={237, 317, 429, 401, 2300, 0};
int PROPERTY_RHO[] ={2702, 19300, 10500, 8933, 3500, 0};


double magnitude(double** B);
void jacobi(double** T, double** Q, double** T_DELTA);
void print_matrix(double** Matrix);
void write_file(double** x,int L_BORDERED);
double getResid(double** T, double** Q);

using namespace std;
 int number_cores;
int main(int argc, char** argv){
  number_cores=stoi(argv[1]);
  int L_BORDERED;
  //printf("Run: %d\n", L);
  L_BORDERED = L + 2;
  int done = 0;
  //double **T, **T_TMP, **B;

  double  resmag,resmago;

  // Allocating the Matrix Rows
  double** T, **T_DELTA, **B;

  T = new double*[L_BORDERED];
  T_DELTA = new double*[L_BORDERED];
  B = new double*[L_BORDERED];

  // Allocating the Matrix Columns
  for(int x = 0; x < L_BORDERED; x++){
    T_DELTA[x] = new double[L_BORDERED]; B[x] = new double[L_BORDERED]; T[x] = new double[L_BORDERED]; T_DELTA[x] = new double[L_BORDERED];
  }

  // Initializing the Values
  for(int x = 0; x < L_BORDERED; x++){
    for(int y = 0; y < L_BORDERED; y++){
      T[x][y] = 273.15; T_DELTA[x][y] = 0.0; B[x][y] = 1000;
    }
  }
 for(int x=0 ; x<L_BORDERED; x++){
   T[x][0]=300;
   T[x][L_BORDERED-1]=300;
   T[0][x]=200;
   T[L_BORDERED-1][x]=300;
 }
 T[(int)((L_BORDERED)/2)-1][(int)((L_BORDERED)/2)-1]=400;
 T[(int)((L_BORDERED)/2)][(int)((L_BORDERED)/2)-1]=400;
 T[(int)((L_BORDERED)/2)-1][(int)((L_BORDERED)/2)]=400;
 T[(int)((L_BORDERED)/2)][(int)((L_BORDERED)/2)]=400;



  #pragma acc data copyin(T[0:L_BORDERED][0:L_BORDERED],B[0:L_BORDERED][0:L_BORDERED],T_DELTA[0:L_BORDERED][0:L_BORDERED])\
  copy(T[0:L_BORDERED][0:L_BORDERED],B[0:L_BORDERED][0:L_BORDERED],T_DELTA[0:L_BORDERED][0:L_BORDERED]) async
  std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();
  for (int totiter=0;totiter<ITER_MAX && done==0;totiter+=RESID_FREQ){

     // do RESID_FREQ jacobi iterations
     jacobi(T, B, T_DELTA);


     write_file(T,L_BORDERED);

    resmag = getResid(T, B)
    ;
    if(totiter==0){
      resmago=resmag+1;
    }
    printf("Iteration: %d %.10f   %.10f\n", totiter, resmago,resmag);
    if( abs(resmago-resmag) < RESID){ done = 1; }

    resmago=resmag;
  }

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

   std::chrono::duration<double> difference_in_time = end_time - begin_time;

   // TIMING LINE 4: Get the difference in seconds.
   double difference_in_seconds = difference_in_time.count();

   // Print the time.
   printf("%d \t %.10f\n", L, difference_in_seconds);
  ofstream file;
  //char* temp = new char[strRank.length() + 1];

  char* file_name = (char*) malloc(15);
  sprintf(file_name, "%d_%d_jacobi_acc_time.txt", number_cores,L);
  file.open(file_name);
  file << "Length of one side"<<L << "\t"<<"Number of nodes"<<number_cores<<"Time taken (s)" << difference_in_seconds<< "\n";
  file.close();

  for(int i = 0; i < L_BORDERED; i++){
     free(T[i]);
     free(B[i]);
     free(T_DELTA[i]);
   }

   free(T);
   free(B);
   free(T_DELTA);
  // Clean up


   return 0;
}

double magnitude(double** B){
   double bmag;
   bmag = 0.0;
   #pragma acc parallel loop reduction(+:bmag) async
   for(int x = 1; x < L+1; x++){
     for(int y = 1; y < L+1; y++){
       bmag  = bmag + B[x][y] * B[x][y];
     }
   }
   #pragma acc wait


   return sqrt(bmag);
}


void jacobi(double** restrict T, double** restrict Q, double** restrict T_DELTA){

   int C_P = PROPERTY_CP[0];
   int K = PROPERTY_K[0];
   int RHO = PROPERTY_RHO[0];
   double qx_conv_left, qx_conv_right, qy_conv_up, qy_conv_down;
   #pragma acc kernels async
      for (int iter=0;iter<RESID_FREQ;iter++){
         // Loop over the inner sections of the grids
        //for(int n = 0; n < number_Of_process+1; n++){
          for(int x=1;x<L+1;x++){
            for(int y=1; y < L+1; y++){
            // Calculating Second Partial Derivative X
              if( !((x==(int)(L+2)/2-1) && (y==(int)(L+2)/2-1))
              || !((x==(int)(L+2)/2-1) && (y==(int)(L+2)/2))
              || !((x==(int)(L+2)/2) && (y==(int)(L+2)/2-1))
              || !((x==(int)(L+2)/2) && (y==(int)(L+2)/2))){
                qx_conv_left = K * (T[x][y-1] - T[x][y]) / D_X;
                qx_conv_right = K * (T[x][y] - T[x][y+1]) / D_X;

                qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
                qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;
                //                 X-Direction Flux           Y-DIrection Flux      Internal Generation
                T_DELTA[x][y] = (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][y]) / (RHO * C_P);
              }
            }
         }
      //  }

         // Copying over values from processor to the main copy.
         for(int x = 1; x < L+1; x++){
           for(int y=0; y < L+1; y++){
             //printf("%d - %d\n", x, y);
             T[x][y] = T[x][y] + T_DELTA[x][y];
           }
         }
      }

}




double getResid(double** restrict T, double** restrict Q){
   int x, y;
   double localres=0,resmag=0;

   int C_P = PROPERTY_CP[0];
   int K = PROPERTY_K[0];
   int RHO = PROPERTY_RHO[0];


   double qx_conv_left, qx_conv_right, qy_conv_up, qy_conv_down;

   //printf("I am rank %d of %d and I received %.8e from the right.\n", my_rank, world_size, right_buffer);
   localres = 0.0;
   resmag = 0.0;

  #pragma acc parallel loop reduction(+:resmag) async
  for(x=1; x < L + 1; x++){
     for(y=1; y < L + 1; y++){

       qx_conv_left = K * (T[x][y-1] - T[x][y]) / D_X;
       qx_conv_right = K * (T[x][y] - T[x][y+1]) / D_X;

       qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
       qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;


       //                 X-Direction Flux           Y-DIrection Flux      Internal Generation
       localres= (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][y]) / (RHO * C_P);
       //RES[x][y] = B[x][y] - T[x][y] + 1.0/4.0 * (T[x+1][y] + T[x-1][y] + T[x][y+1] + T[x][y-1]);
       resmag = resmag + localres * localres;
     }
   }
   #pragma acc wait

   //printf("I am rank %d of %d and I have a local square residual %.8e.\n", my_rank, world_size, resmag);

   //print_matrix(RES, L + 2, size);


   return sqrt(resmag);
}


void print_matrix(double** Matrix){
  int i, j;
  for(i=0;i< L+2; i++){
    for(j=0;j< L+2;j++){
      printf("%.3f ", Matrix[i][j]);
    }
    printf("\n");
  }
}


void write_file(double** x,int L_BORDERED){
  int i, j;
  ofstream file;
  //char* temp = new char[strRank.length() + 1];

  char* file_name = (char*) malloc(15);
  sprintf(file_name, "%d_%d_Heatmap_acc.dat",number_cores, L);


  file.open(file_name);
  for(i=0; i < L_BORDERED; i++){
    for(j=0; j < L_BORDERED; j++){
      file << i << "\t" << j <<"\t" <<x[i][j] << "\n";
    }
  }
  file.close();
}
