// Based on the tutorial from mpitutorial.com/tutorials/mpi-hello-world/
#include <mpi.h>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>

// Maximum number of iterations
#define ITER_MAX 10000000

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

// Useful globals
int world_size; // number of processes
int my_rank; // my process number

int OPTION = 3;


double magnitude(double** B, const int size);
void jacobi(double** T, double** b, double** tmp, const int size);
void update_hot_spot(double** T, int local_size);
void print_matrix(double** T, int x_size, int y_size);
void write_file(double** T, int x_size, int y_size, int standard_size);
double getResid(double** T, double** b, const int size);

using namespace std;

int main(int argc, char** argv)
{

  // Initialize MPI
  MPI_Init(&argc, &argv);

  if(argc >= 1){
    OPTION = atoi(argv[1]);
    printf("Arg %d\n", OPTION);
  }


  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int L_BORDERED;
  //printf("Run: %d\n", L);
  L_BORDERED = L + 2;
  int i,totiter;
  int done = 0;
  //double **T, **T_TMP, **B;

  double bmag, resmag, previous_res;
  int local_size, standard_size;

  // Figure out my local size. The last rank gets the leftover.
  local_size = L_BORDERED/world_size;
  standard_size = local_size;

  if (my_rank == (world_size-1)) { local_size += (L % world_size) ; }

  //printf("I am rank %d of %d and I have a local size %d.\n", my_rank, world_size, local_size);
  // Allocating the Temperature Grid
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

  if(my_rank==0){
    for(x=0; x < L_BORDERED; x++){
      T[x][0] = 300;
    }
  }

  if(my_rank==world_size-1){
    for(x=0; x < L_BORDERED; x++){
      T[x][local_size-1] = 300;
    }
  }



//  printf("Allocated\n");
  // The source only lives on a particular rank!
  int source_right = (L_BORDERED/2)/(L_BORDERED/world_size);
  int source_left = source_left-1;
  // 66 / 2 = 33    //    66 / 4 = 16

  int offset;
  if(my_rank == source_left){
    if(world_size > 1){
      offset = local_size-1;
    }else{
      offset = local_size/2+1;
    }
    T[L_BORDERED/2][offset] = 400;
    T[L_BORDERED/2+1][offset] = 400;
  }else if(my_rank == source_right){
    if(world_size > 1){
      offset = 0;
    }else{
      offset = local_size/2;
    }
    T[L_BORDERED/2][offset] = 400;
    T[L_BORDERED/2+1][offset] = 400;
  }



  bmag = magnitude(B, local_size);
  write_file(T, L_BORDERED, local_size, standard_size);

  std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();


  for (totiter=RESID_FREQ;totiter<ITER_MAX && done==0;totiter+=RESID_FREQ)
  {

     // do RESID_FREQ jacobi iterations
     jacobi(T, B, T_DELTA, local_size);

     write_file(T, L_BORDERED, local_size, standard_size);

     previous_res = resmag;

     resmag = getResid(T, B, local_size);

     if (my_rank == 0) {

       printf("\nIteration: %d - %.10f - %.10f\n", totiter, resmag, abs(resmag-previous_res));
       //print_matrix(T, L+2, local_size);

     }
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



  MPI_Finalize();

   return 0;
}

double magnitude(double** B, const int size)
{
   int i;
   double bmag;
   double global_bmag; // used for global reduce!
   const int lower_limit = (my_rank == 0) ? 1 : 0;
   const int upper_limit = (my_rank == world_size-1) ? size-1 : size;

   i = 0;
   bmag = 0.0;
   global_bmag = 0.0;

   for(int x = 1; x < L + 1; x++){
     for(int y = lower_limit; y < upper_limit; y++){
       bmag  = bmag + B[x][y] * B[x][y];
     }
   }

   // Reduce.
   MPI_Allreduce(&bmag, &global_bmag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   return sqrt(global_bmag);
}









void jacobi(double** T, double** Q, double** T_DELTA, const int size)
{
   int iter,i, x, y;

   // Prepare for async send/recv
   MPI_Request request[4];
   int requests;
   MPI_Status status[4];

   int lower_limit = (my_rank == 0) ? 1 : 0;
   int upper_limit = (my_rank == world_size-1) ? size-1 : size;

   // grab the left and right buffer.
   double* send_left = new double[L+2];
   double* send_right = new double[L+2];

   double* recv_left = new double[L+2];
   double* recv_right = new double[L+2];

   int left_offset = (my_rank==0) ? 1 : 0;
   int right_offset = (my_rank==world_size-1) ? 2 : 1;

   iter = 0; i = 0;
   int count = 0;

   int C_P = PROPERTY_CP[OPTION];
   int K = PROPERTY_K[OPTION];
   int RHO = PROPERTY_RHO[OPTION];

   {
      for (iter=0;iter<RESID_FREQ;iter++)
      {
         requests=0;
         for(i=0; i < L + 2; i++){
           send_left[i] = T[i][left_offset];
           send_right[i] = T[i][size-right_offset];
         }

         // Fill the left buffer. Send to the right, listen from the left.
         MPI_Isend(send_right,   L+2, MPI_DOUBLE, (my_rank+1)%world_size, 0, MPI_COMM_WORLD, request + requests++);
         MPI_Irecv(recv_left, L+2, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 0, MPI_COMM_WORLD, request + requests++);

         // Fill the right buffer. Send to the left, listen from the right.
         MPI_Isend(send_left,   L+2, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 1, MPI_COMM_WORLD, request + requests++);
         MPI_Irecv(recv_right, L+2, MPI_DOUBLE, (my_rank+1)%world_size, 1, MPI_COMM_WORLD, request + requests++);

         double qx_conv_left, qx_conv_right, qy_conv_up, qy_conv_down, delta_T;

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

         // Wait for async.
         MPI_Waitall ( requests, request, status );

         // Doing the Rightmost Part, so use recv_right
         if(my_rank != world_size-1){
           for(x=1; x < L + 1; x++){

             qx_conv_left = K * (T[x][size-2] - T[x][size-1]) / D_X;
             qx_conv_right = K * (T[x][size-1] - recv_right[x]) / D_X;

             qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
             qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;


             T_DELTA[x][size-1] = (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][size-1]) / (RHO * C_P);

           }
         }

         // Doing the Leftmost Part, so use recv_left
         if(my_rank!=0){
           for(x=1; x < L + 1; x++){

             qx_conv_left = K * (recv_left[x] - T[x][0]) / D_X;
             qx_conv_right = K * (T[x][0] - T[x][1]) / D_X;

             qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
             qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;

             T_DELTA[x][0] = (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][0]) / (RHO * C_P);

           }
         }



         // Copying over values from processor to the main copy.
         for(x = 1; x < L+1; x++){
           for(y=lower_limit; y < upper_limit; y++){
             //printf("%d - %d\n", x, y);
             T[x][y] = T[x][y] + T_DELTA[x][y];
           }
         }
         update_hot_spot(T, size);

      }
   }
   MPI_Barrier(MPI_COMM_WORLD);
   free(recv_left); free(recv_right);
}




double getResid(double** T, double** Q, const int size)
{
   int i, x, y;
   double localres=0,resmag=0;
   double global_resmag;

   // Prepare for async send/recv
   MPI_Request request[4];
   int requests;
   MPI_Status status[4];

   double* send_left = new double[L+2];
   double* send_right = new double[L+2];

   int lower_limit = (my_rank == 0) ? 1 : 0;
   int upper_limit = (my_rank == world_size-1) ? size-1 : size;

   double* recv_left = new double[L+2];
   double* recv_right = new double[L+2];


   //printf("Creating sending buffers\n");

   int left_offset = (my_rank==0) ? 1 : 0;
   int right_offset = (my_rank==world_size-1) ? 2 : 1;

   for(i=0; i < L + 2; i++){
     send_left[i] = T[i][left_offset];
     send_right[i] = T[i][size-right_offset];
   }

   int C_P = PROPERTY_CP[OPTION];
   int K = PROPERTY_K[OPTION];
   int RHO = PROPERTY_RHO[OPTION];


   requests=0;

   double qx_conv_left, qx_conv_right, qy_conv_up, qy_conv_down, delta_T;


   // Fill the left buffer. Send to the right, listen from the left.
   MPI_Isend(send_right,   L+2, MPI_DOUBLE, (my_rank+1)%world_size, 1, MPI_COMM_WORLD, request + requests++);
   MPI_Irecv(recv_left, L+2, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 1, MPI_COMM_WORLD, request + requests++);

   ///printf("I am rank %d of %d and I received %f from the left.\n", my_rank, world_size, recv_left[0]);

   // Fill the right buffer. Send to the left, listen from the right.
   MPI_Isend(send_left,   L+2, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 0, MPI_COMM_WORLD, request + requests++);
   MPI_Irecv(recv_right, L+2, MPI_DOUBLE, (my_rank+1)%world_size, 0, MPI_COMM_WORLD, request + requests++);

   //printf("I am rank %d of %d and I received %.8e from the right.\n", my_rank, world_size, right_buffer);

   i = 0;
   localres = 0.0;
   global_resmag = 0.0;
   resmag = 0.0;
   for(x=1; x < L + 1; x++){
     for(y=1; y < size-1; y++){

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

   // Wait for async.
   MPI_Waitall ( requests, request, status );



   // Doing Rightmost Part
   if(my_rank != world_size-1){
     for(x=1; x < L + 1; x++){
       qx_conv_left = K * (T[x][size-2] - T[x][size-1]) / D_X;
       qx_conv_right = K * (T[x][size-1] - recv_right[x]) / D_X;

       qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
       qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;


       localres = (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][size-1]) / (RHO * C_P);
      //localres = B[x][size-1] - T[x][size-1] + 1.0/4.0 * (T[x+1][size-1] + T[x-1][size-1] + T[x][size-2] + recv_right[x]);
       resmag = resmag + localres * localres;
     }
   }



   // Doing the Leftmost Part, so use recv_left
   if(my_rank!=0){
     for(x=1; x < L + 1; x++){

       qx_conv_left = K * (recv_left[x] - T[x][0]) / D_X;
       qx_conv_right = K * (T[x][0] - T[x][1]) / D_X;

       qy_conv_up = K* (T[x][y] - T[x-1][y]) / D_Y;
       qy_conv_down = K* (T[x+1][y] - T[x][y]) / D_Y;

       localres = (qx_conv_left - qx_conv_right +  qy_conv_down - qy_conv_up + Q[x][0]) / (RHO * C_P);
       //localres = B[x][0] - T[x][0] + 1.0/4.0 * (T[x+1][0] + T[x-1][0] + T[x][1] + recv_left[x]);
       resmag = resmag + localres * localres;
     }
   }

   //printf("I am rank %d of %d and I have a local square residual %.8e.\n", my_rank, world_size, resmag);

   //print_matrix(RES, L + 2, size);

   // Reduce.


   MPI_Allreduce(&resmag, &global_resmag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   free(send_left);
   free(send_right);
   free(recv_left);
   free(recv_right);

   return sqrt(global_resmag);
}

void update_hot_spot(double** T, int local_size){

  int source_right = ((L+2)/2)/((L+2)/world_size);
  int source_left = source_left-1;
  // 66 / 2 = 33    //    66 / 4 = 16

  int offset;
  if(my_rank == source_left){
    if(world_size > 1){
      offset = local_size-1;
    }else{
      offset = local_size/2+1;
    }
    T[(L+2)/2][offset] = 400;
    T[(L+2)/2+1][offset] = 400;
  }else if(my_rank == source_right){
    if(world_size > 1){
      offset = 0;
    }else{
      offset = local_size/2;
    }
    T[(L+2)/2][offset] = 400;
    T[(L+2)/2+1][offset] = 400;
  }
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


void write_file(double** x, int x_size, int y_size, int standard_size){
  int i, j;
  ofstream file;

  string strRank = to_string(my_rank);
  //char* temp = new char[strRank.length() + 1];

  char* file_name = (char*) malloc(15);
  sprintf(file_name, "%s_Heatmap.dat", strRank.c_str());


  file.open(file_name);
  for(i=0; i < x_size; i++){
    for(j=0; j < y_size; j++){
      file << i << "\t" << j + my_rank * standard_size - 1 * (my_rank) << "\t" << x[i][j] << "\n";
    }
  }
  file.close();
}
