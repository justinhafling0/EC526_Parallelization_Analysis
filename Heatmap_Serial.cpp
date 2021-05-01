// Based on the tutorial from mpitutorial.com/tutorials/mpi-hello-world/
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>

// Maximum number of iterations
#define ITER_MAX 1000000

// How often to check the relative residual
#define RESID_FREQ 100

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

// Useful globals
int world_size; // number of processes
int my_rank; // my process number

int OPTION = 3;

void jacobi(double** T, double** b, double** tmp, const int size);
void update_hot_spot(double** T, int local_size);
void print_matrix(double** T, int x_size, int y_size);
void write_file(double** T, int x_size, int y_size);
double getResid(double** T, double** b, const int size);

using namespace std;

int main(int argc, char** argv)
{
  if(argc >= 1){
    OPTION = atoi(argv[1]);
    printf("Arg %d\n", OPTION);
  }


  int L_BORDERED;
  //printf("Run: %d\n", L);
  L_BORDERED = L + 2;
  int totiter;
  int done = 0;
  //double **T, **T_TMP, **B;

  double resmag, previous_res;
  int local_size;

  // Figure out my local size. The last rank gets the leftover.
  local_size = L_BORDERED;

  int x, y;

  // Allocating the Matrix Rows
  double** T, **T_DELTA, **B;

  T = new double*[L_BORDERED];
  T_DELTA = new double*[L_BORDERED];
  B = new double*[L_BORDERED];

  // Allocating the Matrix Columns
  for(x = 0; x < L_BORDERED; x++){
    T_DELTA[x] = new double[local_size]; B[x] = new double[local_size]; T[x] = new double[local_size]; T_DELTA[x] = new double[local_size];
  }

  // Initializing the Values
  for(x = 0; x < L_BORDERED; x++){
    for(y = 0; y < local_size; y++){
      T[x][y] = 273.15; T_DELTA[x][y] = 0.0; B[x][y] = 0;
    }
  }

  for(y=0; y < local_size; y++){
    T[0][y] = 200;
    T[L_BORDERED-1][y] = 300;
  }

  for(x=0; x < L_BORDERED; x++){
    T[x][0] = 300;
  }

  for(x=0; x < L_BORDERED; x++){
    T[x][local_size-1] = 300;
  }


  T[L_BORDERED/2][local_size/2] = 400;
  T[L_BORDERED/2][local_size/2+1] = 400;

  T[L_BORDERED/2+1][local_size/2] = 400;
  T[L_BORDERED/2+1][local_size/2+1] = 400;

  std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();


  for (totiter=RESID_FREQ;totiter<ITER_MAX && done==0;totiter+=RESID_FREQ)
  {
     // do RESID_FREQ jacobi iterations
     jacobi(T, B, T_DELTA, local_size);
     write_file(T, L_BORDERED, local_size);

     previous_res = resmag;
     resmag = getResid(T, B, local_size);
     printf("\nIteration: %d - %.10f - %.10f\n", totiter, resmag, abs(resmag-previous_res));

    if (abs(resmag-previous_res) < RESID) { done = 1; }
  }

  if(my_rank==0){
    std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();

   std::chrono::duration<double> difference_in_time = end_time - begin_time;

   // TIMING LINE 4: Get the difference in seconds.
   double difference_in_seconds = difference_in_time.count();

   // Print the time.
   printf("Time: \t %.10f", difference_in_seconds);
   printf("Clock Cycles \t %.10f\n", difference_in_seconds * CLOCKSPEED);
  }


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



void jacobi(double** T, double** Q, double** T_DELTA, const int size)
{
   int iter, x, y;

   iter = 0;

   int C_P = PROPERTY_CP[OPTION];
   int K = PROPERTY_K[OPTION];
   int RHO = PROPERTY_RHO[OPTION];

   {
      for (iter=0;iter<RESID_FREQ;iter++)
      {

         double qx_conv_left, qx_conv_right, qy_conv_up, qy_conv_down;

         // Loop over the inner sections of the grids
         for(x = 1; x < L+1; x++){
           for(y=1; y < size-1; y++){

                                    // Calculating Second Partial Derivative X

            qx_conv_left = K * (T[x][y-1] - T[x][y]) / D_X;
            qx_conv_right = K * (T[x][y] - T[x][y+1]) / D_X;

            qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
            qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;


            //                 X-Direction Flux           Y-DIrection Flux      Internal Generation
            T_DELTA[x][y] = (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][y]) / (RHO * C_P);

            }
         }

         // Copying over values from processor to the main copy.
         for(x = 1; x < L+1; x++){
           for(y=1; y < L+1; y++){
             //printf("%d - %d\n", x, y);
             T[x][y] = T[x][y] + T_DELTA[x][y];
           }
         }
         update_hot_spot(T, size);
      }
   }

}




double getResid(double** T, double** Q, const int size)
{
   int x, y;
   double localres=0,resmag=0;

   int C_P = PROPERTY_CP[OPTION];
   int K = PROPERTY_K[OPTION];
   int RHO = PROPERTY_RHO[OPTION];

   double qx_conv_left, qx_conv_right, qy_conv_up, qy_conv_down;


   localres = 0.0;
   resmag = 0.0;
   for(x=1; x < L + 1; x++){
     for(y=1; y < size-1; y++){

       qx_conv_left = K * (T[x][y-1] - T[x][y]) / D_X;
       qx_conv_right = K * (T[x][y] - T[x][y+1]) / D_X;

       qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
       qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;

       //                 X-Direction Flux           Y-DIrection Flux      Internal Generation
       localres= (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][y]) / (RHO * C_P);
       resmag = resmag + localres * localres;
     }
   }

   return sqrt(resmag);
}

void update_hot_spot(double** T, int local_size){


    T[(L+2)/2][local_size/2] = 400;
    T[(L+2)/2][local_size/2+1] = 400;

    T[(L+2)/2+1][local_size/2] = 400;
    T[(L+2)/2+1][local_size/2+1] = 400;
}


void print_matrix(double** Matrix, int x_size, int size){
  int i, j;
  for(i=0;i< x_size; i++){
    for(j=0;j< size;j++){
      printf("%.3f ", Matrix[i][j]);
    }
    printf("\n");
  }
}


void write_file(double** x, int x_size, int y_size){
  int i, j;
  ofstream file;

  string strRank = to_string(my_rank);
  //char* temp = new char[strRank.length() + 1];

  char* file_name = (char*) malloc(15);
  sprintf(file_name, "%s_Heatmap.dat", strRank.c_str());


  file.open(file_name);
  for(i=0; i < x_size; i++){
    for(j=0; j < y_size; j++){
      file << i << "\t" << j << "\t" << x[i][j] << "\n";
    }
  }
  file.close();
}
