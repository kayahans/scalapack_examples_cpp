#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "mpi.h"

/**
 * Compile with 
 * mpicxx example_pdsyev.cpp -L/usr/local/Cellar/scalapack/2.1.0_2/lib -lscalapack
 */

extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern "C" void blacs_gridexit_(int*);
extern "C" int numroc_(int*, int*, int*, int*, int*);
extern "C" void pdsyev_(char *jobz, char *uplo, int *n, double *a, int *ia, int *ja,
                 int *desca, double *w, double *z, int *iz, int *jz, int *descz,
                 double *work, int *lwork, int *info);
extern "C" void pdelset_(double* mat, const int* i, const int* j, const int* desc, const double* a);

void read_square_matrix(double *A, int n) {
    // Create any matrix using MATLAB as below
    // M = magic(1000) -> here M is a 1000x1000 matrix
    // M = (M+M')/2; -> make it symmetric
    // eig(M) -> print eig for comparison
    // writematrix(M, 'M.txt') -> writes to M.txt
    // MATLAB writes row major
    std::fstream newfile;
    newfile.open("M.txt", std::ios::in);
    int a_ind = 0;
    if (newfile.is_open()) {
        std::string tp;
        while (getline(newfile, tp)) {
            std::stringstream ss(tp);
            std::string intermediate;
            while (getline(ss, intermediate, ',')) {
                // printf("%d, %f\n", a_ind, std::stod(intermediate));
                A[a_ind] = std::stod(intermediate);
                a_ind++;
            }
        }
        newfile.close();  // close the file object.
    }
    // printf("Last element of sq matrix %f\n",A[n*n-1]);
}

int main(int argc, char **argv) {
    int izero = 0;
    int ione  = 1;
    int myrank_mpi, nprocs_mpi;
    /**
     *  User defined values
     **/
    int n       = 55;       // (Global) Matrix size
    int nprow   = 2;      // Number of row procs
    int npcol   = 2;      // Number of column procs
    int nb      = 25;       // (Global) Block size
    char jobz   = 'V';
    char uplo   = 'U';      // Matrix is lower triangular
    char layout = 'R';    // Block cyclic, Row major processor mapping
    // User defined values end

    double *input_mat;
    input_mat = new double[n*n];

    // currently the matrix is read by all the MPI processes, this is inefficient
    read_square_matrix(input_mat, n);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Matrix is read on proc %d/%d for MPI\n",myrank_mpi,nprocs_mpi);

    assert(nprow * npcol == nprocs_mpi);

    // Initialize BLACS
    int iam, nprocs;
    int zero = 0;
    int ictxt, myrow, mycol;
    blacs_pinfo_(&iam, &nprocs) ; // BLACS rank and world size
    blacs_get_(&zero, &zero, &ictxt ); // -> Create context
    blacs_gridinit_(&ictxt, &layout, &nprow, &npcol ); // Context -> Initialize the grid
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol ); // Context -> Context grid info (# procs row/col, current procs row/col)

    // Compute the size of the local matrices
    int mpA    = numroc_( &n, &nb, &myrow, &izero, &nprow ); // My proc -> row of local A
    int nqA    = numroc_( &n, &nb, &mycol, &izero, &npcol ); // My proc -> col of local A

    printf("Proc %d/%d for MPI, proc %d/%d for BLACS in position (%d,%d)/(%d,%d) with local matrix %dx%d, global matrix %d, block size %d\n",myrank_mpi,nprocs_mpi,iam,nprocs,myrow,mycol,nprow,npcol,mpA,nqA,n,nb);

    double *A = (double *)calloc(mpA*nqA,  sizeof(double)) ;
    if (A==NULL){ printf("error of memory allocation A on proc %dx%d\n", myrow, mycol); exit(0); }
    
    
    double *Z = (double *)calloc(mpA*nqA,  sizeof(double)) ;
    if (Z==NULL){ printf("error of memory allocation VT on proc %dx%d\n", myrow, mycol); exit(0); }
    
    int min_mn = n;
    double *W = (double *)calloc(min_mn,  sizeof(double)) ;
    if (W==NULL){ printf("error of memory allocation S on proc %dx%d\n", myrow, mycol); exit(0); }

    int k = 0;
    printf("Proc %d/%d for MPI, proc %d/%d for BLACS at %d\n", myrank_mpi,nprocs_mpi,iam,nprocs, nb*(nb*myrow*npcol + mycol));

    // Create descriptor
    int descA[9];
    int descZ[9];
    int info;
    int lddA = mpA > 1 ? mpA : 1;
    descinit_( descA,  &n, &n, &nb, &nb, &izero, &izero, &ictxt, &lddA, &info );
    descinit_( descZ,  &n, &n, &nb, &nb, &izero, &izero, &ictxt, &lddA, &info );
    if(info != 0) {
        printf("Error in descinit, info = %d\n", info);
    }
    
    // Below here works for 2x2 proc useful for understanding
    // int start_idx = myrow*nb + mycol*nb*n;
    // for (int j = 0; j < nqA; j++) { // local col
    //     for (int i = 0; i < mpA; i++) { // local row
    //         A[k] = input_mat[i+j*n+start_idx];
    //         k++;
    //     }
    // }
    
    // Pdelset routine is easier and general for any block cyclic distribution
    for (int i=1;i<=n;i++) {
        for (int j=1;j<=n;j++) {
            // Read row major matrix
            pdelset_(A, &i, &j, descA, &(input_mat[(j-1)+(i-1)*n]));
        }
    }

    double *work = (double *)calloc(2,sizeof(double)) ;
    if (work==NULL){ printf("error of memory allocation for work on proc %dx%d (1st time)\n",myrow,mycol); exit(0); }
    int lwork=-1;
    pdsyev_( &jobz, &uplo, &n, A, &ione, &ione, descA, W, Z, &ione, &ione, descZ, work, &lwork, &info );

    lwork= (int) work[0];
    free(work);
    work = (double *)calloc(lwork,sizeof(double)) ;
    if (work==NULL){ printf("error of memory allocation work on proc %dx%d\n",myrow,mycol); exit(0); }

    double MPIt1 = MPI_Wtime();
    pdsyev_( &jobz, &uplo, &n, A, &ione, &ione, descA, W, Z, &ione, &ione, descZ, work, &lwork, &info );

    double MPIt2 = MPI_Wtime();
    double MPIelapsed=MPIt2-MPIt1;

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank_mpi == 0) {
        printf("[DIAGONALIZER] n=%d\t(%d,%d)\t%d\tjobz=%c\t%8.2fs \n",n,nprow,npcol,nb,jobz,MPIelapsed);
        for (int i = 0; i < n; i++) {
            printf("%f\n", W[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    free(work);
    free(W);
    free(Z);
    free(A);
    free(input_mat);

    // Exit and finalize
    blacs_gridexit_(&ictxt);
    MPI_Finalize();

    return 0;
}
