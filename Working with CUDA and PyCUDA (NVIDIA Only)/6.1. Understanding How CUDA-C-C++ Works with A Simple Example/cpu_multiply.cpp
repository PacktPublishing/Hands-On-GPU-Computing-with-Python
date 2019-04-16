#include <iostream> //Defining standard input/output stream objects
#include <math.h>   //For using predefined math functions
#define N 500000000 //500 Million elements

clock_t begin, end;
float cpu_time_used;
     
// This is a function to multiply two array elements and also update the results on the second array
void multiply(int n, double *p, double *q)
{
  for (int i = 0; i < n; i++)
      q[i] = p[i] * q[i];
}

int main(void)
{
  double *p = new double[N];
  double *q = new double[N];

  // initialize arrays p and q on the host
  for (int i = 0; i < N; i++) {
    p[i] = 24.0;
    q[i] = 12.0;
  }

  // Run function on 500 Million elements on the CPU
  begin = clock();
  multiply(N, p, q);
  end = clock();
  cpu_time_used = ((double) (end - begin)) / CLOCKS_PER_SEC;

  // Verifying all values to be 288.0
  // fabs(q[i]-288) (absolute value) should be 0
  double maxError = 0.0;
  for (int i = 0; i < N; i++){
      maxError = fmax(maxError, fabs(q[i]-288.0));
  }
      std::cout << "Multiply function CPU execution time: " << cpu_time_used << " second(s)" << std::endl;
      std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] p;
  delete [] q;

  return 0;
}

