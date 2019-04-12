## Searching, Sorting and Divide and Conquer
### Insertion Sort
Insertion sort is an efficient algorithm for sorting a small number of elements. In-place sorting. Insertion sort iterates one input element in each repetition, and growing a sorted output list. At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list, and inserts it there. It repeats until no input elements remain. https://en.wikipedia.org/wiki/Insertion_sort 

#### Pseudocode
```
Insert-sort(A[1..N])
for j = 2 to A.length
    key = A[j]
    i = j - 1 //pointer to traverse left side of the collection
    while i > 0 AND A[i] > key
        A[i+1] = A[i]
        i = i -1
        A[i+1] = key
```
Running time: in worst case scenario, 4-7 would iterates n times and line 1 iterates n for all elements. So, O(n^2) is the running time and space is O(1).

```java
public static void sort(int[] arr) {
	if (arr == null)
		return;

	for(int i = 1; i < arr.length; i++) {
		int key = arr[i];
		/** pointer to traverse left side of the collection */
		int j = i - 1;

		/** traverse left side and compare each element and place into right position */
		while (j >= 0 && arr[j] > key) {
			arr[j + 1] = arr[j];
			j--;
			arr[j + 1] = key;
		}
	}
}
```

### Recursive Insertion Sort
In order to sort `A[0..n-1]` we recursively sort `A[0..n-2]` first and then insert `A[n-1]` into the sorted array of `A[0..n-2]`.

#### Pseudocode
```
Recursive_Insertion(A, n)
1. if n > 1
2.    Recursive_Insertion(A, n - 1)
3.    insert(A, n)

insert(A, i):
/** exactly follow the routine for iterative insertion sort */
1. key = A[i]
2. j = i - 1
3. while j >= 0 AND A[j] > key
4.    A[j + 1] = A[j]; j--;
5.    A[j + 1] = key
```
#### Java Code
```java
public class RecursiveInsertionSort {
	public static void recursiveInsertionSort(int[] arr, int n) {
		if (n > 0) {
			recursiveInsertionSort(arr, n - 1);
			insert(arr, n);
		}
	}

	public static void insert(int[] arr, int i) {
		int key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j] > key) {
			arr[j + 1] = arr[j];
			j--;
			arr[j + 1] = key;
		}
	}
}
```
Running time is same as iterative one i.e `O(n^2)` and it would take system space due to recursive call.


### Add Two N-Bit Binary Number
Consider the problem of adding two n-bit binary integers, stored in two n-element arrays A & B. The sum of the integers should be stored in binary form in an (n+1) element array C. State the problem formally and write pseudo code for adding the two integers.

https://stackoverflow.com/questions/5610024/please-explain-to-me-the-solution-for-the-problem-below

```java
public static int[] addTwoBinary(int[] a, int[] b) {
	if (a.length != b.length)
		return null;

	int[] c = new int[a.length + 1];
	Arrays.fill(c, 0);

	for(int i = a.length - 1; i >= 0; i--) {
		int sum = a[i] + b[i] + c[i + 1];
		/** store the sum into the current position and put carry into previous place */
		c[i + 1] = sum % 2;
		c[i] = sum / 2;
	}
	
	return c;
}
```
Time and space complexity - O(N)

### Merge Sort
Many useful algorithms are recursive in structure: to solve a given problem, they call themselves recursively one or more times to deal with closely related subproblems. These algorithms typically follow a divide-and-conquer approach: they break the problem into several subproblems that are similar to the original problem but smaller in size, solve the subproblems recursively, and then combine these solutions to create a solution to the original problem.
<p>
The divide-and-conquer paradigm involves three steps at each level of the recursion: Divide the problem into a number of subproblems that are smaller instances of the same problem. Conquer the subproblems by solving them recursively. If the subproblem sizes are small enough, however, just solve the subproblems in a straightforward manner. Combine the solutions to the subproblems into the solution for the original problem. The merge sort algorithm closely follows the divide-and-conquer paradigm.
</p>

- Divide: Divide the n-element sequence to be sorted into two subsequences of  elements each (takes  to split)
- Conquer: Sort the two subsequences recursively using merge sort.
- Combine: Merge the two sorted subsequences to produce the sorted answer (takes  to merge)

The condition to stop the recursion process is when the sequence to be sorted has length 1. https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/tutorial/ 

#### Pseudocode:
```
Merge-Combine(A, p, q, r)
-------------------------
1. Create left and right temp array of size L(q - p + 1), R(r - q)
//Copy the collection data into temp array
2. for i = 0 to (q - p + 1)
3.     L[i] = A[start + i]
4. for j = 0 to (r - q)
5.     R[j] = A[q + j]
6. i = 0, j = 0, k = 0
//compare each sorted list and choose the smaller number
7. while i < (q - p + 1) AND j < (r - q) AND k < A.length
8.     if L[i] < R[j]
9.        A[start + k] = L[i]; i++;
10.    else
11.       A[start + k] = R[j]; j++
12.    k++;
//if any sorted list is already exhausted, then copy rest of the element from the other list
13. while i < (q - p + 1)
14.    A[start + k] = L[i]; i++; k++;
15. while j < (r - q)
16.    A[start + k] = R[j]; j++; k++;

Merge-Sort(A, p, r)
-------------------
1. if p < r
2.    mid = (p + r) / 2
3.    Merge-Sort(A, p, mid)
4.    Merge-Sort(A, mid + 1, r)
5.    Merge-Combine(A, p, mid, r)
```
#### Java Code

```java
import java.util.*;

public class MergeSort {
	public static void merge(int[] arr, int start, int mid, int end) {
		int[] left = new int[mid - start + 1];
		int[] right = new int[end - mid];

		/** Copy the collection data into temp array */
		for (int i = 0; i < left.length; i++)
			left[i] = arr[start + i];

		for (int i = 0; i < right.length; i++)
			right[i] = arr[mid + 1 + i];

		/** compare each sorted list and choose the smaller number */
		int i = 0, j = 0, k = 0;
		while (i < left.length && j < right.length && k < arr.length) {
			if (left[i] < right[j])
				arr[start + k] = left[i++];
			else
				arr[start + k] = right[j++];
			k++;
		}

		/** if any sorted list is already exhausted, then copy rest of the element from the other list */
		while (i < left.length) {
			arr[start + k] = left[i++];
			k++;
		}

		while (j < right.length) {
			arr[start + k] = right[j++];
			k++;
		}
		
		System.out.println(Arrays.toString(arr));
	}
	
	/** Main Sub-Routine */
	public static void mergeSort(int[] arr, int start, int end) {
		if (start < end) {
			int mid = (start + end) / 2;
			mergeSort(arr, start, mid);
			mergeSort(arr, mid + 1, end);
			merge(arr, start, mid, end);	
		}
	}
}
```
#### Running Time analysis
When an algorithm contains a recursive call to itself, we can often describe its running time by a recurrence equation or recurrence, which describes the overall running time on a problem of size n in terms of the running time on smaller inputs. We can then use mathematical tools to solve the recurrence and provide bounds on the performance of the algorithm.

Let, T(n) be the running time on a problem of size n. If the problem size is very small like n << c, then T(n) ~ O(1).
Otherwise, suppose we divide the problem set into ```a``` subproblems each with size of ```n/b```. (For merge sort a = b = 2)

Now, for n size the running time is T(n)
     for n/b size the running time would be T(n/b)

So, we have ```a``` subproblems, totalling ```a.T(n/b)``` to solve ```a``` subproblems.

If we take D(n) time divide the problem into subproblems and C(n) time to combine the solutions, then
```T(n) = a.T(n/b) + D(n) + C(n)```

For specific to merge sort, the divide step i.e. calculating the mid value of the collection takes contant time, i.e. D(n) = O(1). Conquer step, we recursively solved two subproblems each of size n/2 which contributes ```2T(n/2)``` running time. The merge-combine procedure takes linear time ~ O(n)

``` T(n) = 2T(n/2) + O(n) when n > 1```

So, as per master theorem, ```T(n) = O(nlogn)```

Space complexity - O(n) except the recursion itself as it will take some system place due to recursion.


### Maximum Sub-Array Problem

Find the sum of the contiguous subarray of numbers which has the largest sum. For example

A {-2,-5,<b>6,-2,-3,1,5</b>,-6} the maximum sum is 7 from the subarray of {6,-2,-3,1,5}

The brute force method is to run two loops. The outer loop picks the beginning element, the inner loop finds the maximum possible sum with first element picked by outer loop and compares this maximum with the overall maximum. Finally return the overall maximum. The time complexity of the Naive method is `O(n^2)`.

`
CLRS uses a different but unique problem: buying and selling stock price. So, if we receive a future stock price data, it is not just to pick maximum selling price and minimum buying price to maximize the profit. Rather we will find the daily changes in price by substracting A[i] - A[i - 1] i.e. difference between two consecutive days stock price. We can now use this dataset as the source for maximum sub-array problem.
`
<p>
Important point to be noted that *The maximum-subarray problem is interesting only when the array contains
some negative numbers. If all the array entries were nonnegative, then the maximum-subarray problem would present no challenge, since the entire array would give the greatest sum.*
</p>

#### Solution using Divide and Conquer method

We divide the original problem into two subarrays of equal size, find the mid point. So we have left sub-array A[low..mid] and right subarray of A[mid+1..high]. Now, the maximum subarray A[i..j] must lie in exactly one of the following places:

- within left sub-array i.e. ```low <= i <= j <= mid```
- within right sub-array i.e. ```mid + 1 <= i <= j <= high```
- crossing the mid-point i.e. ```low <= i <= mid <= j <= high```

We can find maximum subarrays of A[low..mid] and A[mid+1..high] recursively, because these two subproblems are smaller instances of the problem of finding a maximum subarray. So, only thing is we need to find the maximum subarray that crosses the midpoint. Now, any subarray crossing the midpoint is itself made of two subarrays A[i..mid] and A[mid+1..j]. Therefore, we just need to find the maximum subarrays of the form A[i..mid] and A[mid+1..j] and combine them.

#### Pseudocode

So the below procedure takes an input A, low, mid and high indices and returns a tuple containing the indices demarcating a maximum subarray that crosses the midpoint along with the sum of the max subarray.

```
Find-Max-Crossing-SubArray(A, low, mid, high):
----------------------------------------------
1. left_sum = -inf, sum = 0
2. for i = mid to low
3.     sum = sum + A[i]
4.     if sum > left_sum
5.        left_sum = sum; max_left = i
6. right_sum = -inf, sum = 0
7. for j = mid+1 to high
8.     sum = sum + A[j]
9.     if sum > right_sum
10.        right_sum = sum; max_right = j
11. return (max_left, max_right, left_sum+right_sum)
```

So, the above procedure calculates the max value in left and right direction from the midpoint. The running time of the above sub-routine is O(n).

With a linear-time *Find-Max-Crossing-SubArray* procedure in hand, we can write pseudocode for a divide-and-conquer algorithm to solve the maximum subarray problem.

```
Find-Max-SubArray(A, low, high):
--------------------------------
1. if low == high
2.    return (low, high, A[high])
3. else mid = (low+high)/2
4.    (left_i, left_j, left_sum) = Find-Max-SubArray(A, low, mid)
5.    (right_i, right_j, right_sum) = Find-Max-SubArray(A, mid+1, high)
6.    (cross_i, cross_j, cross_sum) = Find-Max-Crossing-SubArray(A, low, mid, high)
7.    if left_sum > right_sum AND left_sum > cross_sum
8.        return (left_i, left_j, left_sum)
9.    else if right_sum > left_sum AND right_sum > cross_sum
10.       return (right_i, right_j, right_sum)
11.   else
12.       return (cross_i, cross_j, cross_sum)
```
#### Java Code

```java
public static int findMaxCrossSubarray(int[] arr, int low, int mid, int high) {
	int left_sum = Integer.MIN_VALUE, right_sum = Integer.MIN_VALUE, sum = 0;
	/** find max value in the left side of the midpoint */
	for (int i = mid; i >= low; i--) {
		sum += arr[i];
		if (sum > left_sum)
			left_sum = sum;
	}

	sum = 0;
	/** find max value in the right side of the midpoint */
	for (int j = mid; j <= high; j++) {
		sum += arr[j];
		if (sum > right_sum)
			right_sum = sum;
	}
	return left_sum + right_sum;
}

/** main sub-routine */
public static int findMaxSubarray(int[] arr, int low, int high) {
	if (low == high)
		return arr[high];
	else {
		int mid = (low + high) / 2;
		return Math.max(Math.max(findMaxSubarray(arr, low, mid),
			            findMaxSubarray(arr, mid + 1, high)),
		                findMaxCrossSubarray(arr, low, mid, high));
	}
}
```

Running time of n element is ```T(n) = 2T(n/2) + O(n)```. This is similar like merge sort as the original is divided into two equal sub-problem in each recursive call. Additional ```O(n)``` time is due the work for crossing the midpoint. So, ```T(n) = O(nlogn)```. Space complexity is contant O(1) as no auxilary space is utilized. However, recursive call is itself taking system space for the call stack.

#### Linear Time Solution for Max Subarray problem (Kadane's Algorithm)

Simple idea of the Kadane’s algorithm is to look for all positive contiguous segments of the array and keep track of maximum sum contiguous segment among all positive segments. Each time we get a positive sum compare it with current max so far and update it if the sum is greater than max_so_far.

#### Pseudocode
```
1. max = 0, sum = 0
2. for i = 0 to A.length - 1
3.     sum = Math.max(A[i], sum + A[i])
4.     max = Math.max(sum, max)
```

As per Kadane's algorithm if we find sum is negative, we will set ```sum = 0```. So, for this set {-2,-5,-6,-2} the above algorithm yields zero rather than true output. We can modify little bit to allow negative set of numbers

```
1. max = A[0], sum = A[0]
2. for i = 1 to A.length - 1
3.     sum = Math.max(A[i], sum + A[i])
4.     max = Math.max(sum, max)
```
#### Java Code
```java
public static int kadane(int[] arr) {
	int max = arr[0], sum = arr[0];
	for (int i = 1; i < arr.length; i++) {
		sum = Math.max(arr[i], sum + arr[i]);
		max = Math.max(max, sum);
	}
	return max;
}
```
Running is O(n) and Space is O(1). Refer: https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/

### Strassen's Algorithm for Matrix Multiplication
If A and B are square matrix of size n then the matrix multiplication C = A.B can be defined as ```c[i][j] = sum of a[i][k].b[k][j]``` where k is 0 to n-1.

<a href="https://www.codecogs.com/eqnedit.php?latex=c_i_j&space;=&space;\sum_{k&space;=&space;0}^{n-1}a_i_k.b_k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_i_j&space;=&space;\sum_{k&space;=&space;0}^{n-1}a_i_k.b_k_j" title="c_i_j = \sum_{k = 0}^{n-1}a_i_k.b_k_j" /></a>

We must compute n^2 matrix entries, and each is the sum of n values. The following procedure takes n ( n matrices A and B and multiplies them, returning their n ( n product C)

#### Pseudocode for the naive approache
```
Square-Matrix-Multiply(A, B):
1. n = A.rows; C = new n*n matrix
2. for i = 1 to n
3.   for j = 1 to n
4.       c[i][j] = 0
5.       c[i][j] = c[i][j] + a[i][k].b[k][j]
6. return C
```
```java
/** naive approach to matrix multiplications */
public static void matrixMul(int[][] a, int[][] b) {
	int n = a.length;
	int[][] c = new int[n][n];

	for(int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i][j] = 0;
			for (int k = 0; k < n; k++)
				c[i][j] += a[i][k]*b[k][j];
		}
	}
}
```
Running time is O(n^3). Now, we will visit Strassen's algorithm which has the running less than the naive

<a href="https://www.codecogs.com/eqnedit.php?latex=O(n^{2.81})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O(n^{2.81})" title="O(n^{2.81})" /></a>

Before Strassen's algorithm, lets understand how the above problem can be solve using the Divide and Conquer approach.

#### Matrix Multiplication using Divide and Conquer

To keep things simple, when we use a divide-and-conquer algorithm to compute the matrix product C = A.B, we assume that ```n``` is an exact power of 2 in each of the matrices. We make this assumption because in each divide step, we will divide n matrices into four ```n/2 x n\2``` by assuming that n is an exact power of 2, we are guaranteed that as long as ```n >= 2```, the dimension ```n\2``` is an integer. We partition (index partition) each of A, B and C into four ```n/2 x n\2```matrices

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}C_1_1&space;&C_1_2&space;&&space;\\&space;C_2_1&space;&&space;C_2_2&space;\end{pmatrix}&space;=&space;\begin{pmatrix}A_1_1&space;&A_1_2&space;&&space;\\&space;A_2_1&space;&&space;A_2_2&space;\end{pmatrix}&space;&plus;&space;\begin{pmatrix}B_1_1&space;&B_1_2&space;&&space;\\&space;B_2_1&space;&&space;B_2_2&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}C_1_1&space;&C_1_2&space;&&space;\\&space;C_2_1&space;&&space;C_2_2&space;\end{pmatrix}&space;=&space;\begin{pmatrix}A_1_1&space;&A_1_2&space;&&space;\\&space;A_2_1&space;&&space;A_2_2&space;\end{pmatrix}&space;&plus;&space;\begin{pmatrix}B_1_1&space;&B_1_2&space;&&space;\\&space;B_2_1&space;&&space;B_2_2&space;\end{pmatrix}" title="\begin{pmatrix}C_1_1 &C_1_2 & \\ C_2_1 & C_2_2 \end{pmatrix} = \begin{pmatrix}A_1_1 &A_1_2 & \\ A_2_1 & A_2_2 \end{pmatrix} + \begin{pmatrix}B_1_1 &B_1_2 & \\ B_2_1 & B_2_2 \end{pmatrix}" /></a>

Rewriting the above equations:

<a href="https://www.codecogs.com/eqnedit.php?latex=\\C_1_1&space;=&space;A_1_1.B_1_1&space;&plus;&space;A_1_2.B_2_1&space;\\C_1_2&space;=&space;A_1_1.B_1_2&space;&plus;&space;A_1_2.B_2_2&space;\\C_2_1&space;=&space;A_2_1.B_1_1&space;&plus;&space;A_2_2.B_2_1&space;\\C_2_2&space;=&space;A_2_1.B_1_2&space;&plus;&space;A_2_2.B_2_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\C_1_1&space;=&space;A_1_1.B_1_1&space;&plus;&space;A_1_2.B_2_1&space;\\C_1_2&space;=&space;A_1_1.B_1_2&space;&plus;&space;A_1_2.B_2_2&space;\\C_2_1&space;=&space;A_2_1.B_1_1&space;&plus;&space;A_2_2.B_2_1&space;\\C_2_2&space;=&space;A_2_1.B_1_2&space;&plus;&space;A_2_2.B_2_2" title="\\C_1_1 = A_1_1.B_1_1 + A_1_2.B_2_1 \\C_1_2 = A_1_1.B_1_2 + A_1_2.B_2_2 \\C_2_1 = A_2_1.B_1_1 + A_2_2.B_2_1 \\C_2_2 = A_2_1.B_1_2 + A_2_2.B_2_2" /></a>

Each of these four equations specifies two multiplications of ```n/2 x n/2``` matrices and their addition. We can use these above equation to build the divide and conquer recursive solution

#### Pseudocode
```
Square_Mat(A, B)
1. n = A.rows; create C as n * n matrix
2. if n == 1
3.    c_11 = a_11.b_11
4. else
5.    c_11 = Square_Mat(A_11, B_11) + Square_Mat(A_12, B_21)
5.    c_12 = Square_Mat(A_11, B_12) + Square_Mat(A_12, B_22)
5.    c_21 = Square_Mat(A_21, B_11) + Square_Mat(A_22, B_21)
5.    c_22 = Square_Mat(A_21, B_12) + Square_Mat(A_22, B_22)
```
Running time T(n) = Time taken by 8 recursive call at each subproblems + Base case + Four matrix addtion O(n^2)
```T(n) = 8T(n/2) + O(1) + O(n^2)``` .By resolving the above equation we have ```T(n) = O(n^3)```.

#### Strassen's method

Strassen's method perform 7 recursive multiplication than 8 recursive multiplication than the simple divide and conquer method. Steps are:

```
1. Divide the input matrix A, B and output matrix C into n/2*n/2 submatrices
2. Recursively compute 7 matrix product P1, P2,..P7. Each matrix Pi is n/2*n/2
	P1 = A11.B12 - A11.B22
	P2 = A11.B22 + A12.B22
	P3 = A21.B11 + A22.B11
	P4 = A22.B21 - A22.B11
	P5 = A11.B11 + A11.B22 + A22.B11 + A22.B22
	P6 = A12.B21 + A12.B22 - A22.B21 - A22.B22
	P7 = A11.B11 + A11.B12 - A21.B11 - A21.B12
3. Compute the desire submatrices C_11, C_12, C_21 and C_22
	C11 = P5 + P4 - P2 + P6
	C12 = P1 + P2
	C21 = P3 + P4
	C22 = P5 + P1 - P3 - P7
```

Running time analysis: Divide the matrix takes O(1), and adding/substracting the matrix takes O(n^2) and 7 recursive call takes ```7.T(n/2)```. So, total ```T(n) = 7T(n/2) + O(n^2)```. By using master theorem, ```T(n) = O(n^2.81)```

Ref: 1) https://www.youtube.com/watch?v=0oJyNmEbS4w 2) https://www.geeksforgeeks.org/strassens-matrix-multiplication/

### Master Theorem for solving recurrence

```T(n) = aT(n/b) + f(n)```

The running time of a recurrence problem: divides a problem of size n into ```a``` subproblems, each of size ```n/b```. The ```a``` subproblems are solved recursively each in time ```T(n/b)```. The function ```f(n)``` encompasses the cost of dividing and combining the result of the subproblems.

So, the master theorem is as follows

```T(n) = aT(n/b) + O(n^c)```

- <a href="https://www.codecogs.com/eqnedit.php?latex=c&space;<&space;log_b&space;{a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c&space;<&space;log_b&space;{a}" title="c < log_b {a}" /></a> 
then T(n) = O(<a href="https://www.codecogs.com/eqnedit.php?latex=n^{log_b{a}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{log_b{a}}" title="n^{log_b{a}}" /></a>)

- <a href="https://www.codecogs.com/eqnedit.php?latex=c&space;=&space;log_b{a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c&space;=&space;log_b{a}" title="c = log_b{a}" /></a> then T(n) = O(<a href="https://www.codecogs.com/eqnedit.php?latex=n^c&space;log&space;{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^c&space;log&space;{n}" title="n^c log {n}" /></a>)

- <a href="https://www.codecogs.com/eqnedit.php?latex=c&space;>&space;log_b&space;{a}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c&space;>&space;log_b&space;{a}" title="c > log_b {a}" /></a> then T(n) = O(n^c)


## Randomized Algorithm

An algorithm that uses random numbers to decide what to do next anywhere in its logic is called Randomized Algorithm. For example, in Randomized Quick Sort, we use random number to pick the next pivot (or we randomly shuffle the array). Typically, this randomness is used to reduce time complexity or space complexity in other standard algorithms.

### Hiring problem

Suppose that you need to hire a new candidate. The below pseudocode expresses the strategy for hiring the candidate that is each candidate will be evaluated and choose the best candidate than the existing one

#### Pseudocode
```
Hiring-Candidate(n)
1. best = 0 //dummy candidate
2. for i = 1 to n
3.     interview candidate i
4.     if i > best
5.        best = i
6.        hire candidate i
```
So, given n candidate, let m be the no. of candidate is hired. So, the total cost (almost similar like running time) is ```O(cost_of_inderview * n + cost_of_hire * m)```. No matter how many people we hire, we always interview n candidates and thus always incur the cost ```O(cost_of_inderview * n)``` associated with interviewing. We therefore concentrate on analyzing ```O(cost_of_hire * m)``` the hiring cost. This quantity ```m``` varies with each run of the algorithm.

In the worst case, we actually hire every candidate that we interview. This situation occurs if the candidates come in strictly increasing order of quality, in which case we hire n times, for a total hiring cost of ```O(cost_of_hire * n)``` 

This scenario serves as a model for a common computational paradigm. We often need to find the maximum or minimum value in a sequence by examining each element of the sequence and maintaining a current “winner.”

##### Q. Find min or max from a sequence
```java
/** find min value */
public static int min(int[] arr) {
	int min = 0;
	for (int i = 0; i < arr.length; i++) {
		if (min > arr[i])
			min = arr[i];
	}
	return min;
}
```
Running time O(n) and space is O(1).

### Probabilistic analysis

Probabilistic analysis is the use of probability in the analysis of problems. Most commonly, we use probabilistic analysis to analyze the running time of an algorithm. In order to perform this analysis, we must know about the distribution of the inputs. Then we analyze our algorithm, computing an *average-case running time*, where we take the average over the distribution of the possible inputs.

In many cases, we know very little about the input distribution. Even if we do know something about the distribution, we may not be able to model this knowledge computationally. Yet we often can use probability and randomness as a tool for algorithm design and analysis, by making the behavior of part of the algorithm random.

In the above hiring problem, it may seem as if the candidates are being presented to us in a random order, we have no way of knowing whether or not they really are. Thus, in order to develop a randomized algorithm for the hiring problem, we must change the model slightly. We say that the employment agency has n candidates, and they send us a list of the candidates in advance. On each day, we choose, randomly, which candidate to interview. 

Instead of relying on a guess that the candidates come to us in a random order, we have instead gained control of the process and enforced a random order. More generally, we call an algorithm randomized if its behavior is determined not only by its input but also by values produced by a **random-number generator**. In practice, most programming environments offer a **pseudorandom-number generator - a deterministic algorithm returning numbers that look statistically random**.

When analyzing the running time of a randomized algorithm, we take the **expectation of the running time** over the distribution of values returned by the random number generator.

#### Indicator Random Variable

Indicator random variables provide a method for converting probabilities and expectations. For sample space ```S``` and an event ```A```, the indicator random variable for {A} defined as 

<a href="https://www.codecogs.com/eqnedit.php?latex=I(A)&space;=&space;\left\{\begin{matrix}1&space;&if&A&occurs&space;&&space;\\&space;0&space;&if&A&doesn't&occur&space;&&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?I(A)&space;=&space;\left\{\begin{matrix}1&space;&if&A&occurs&space;&&space;\\&space;0&space;&if&A&doesn't&occur&space;&&space;\end{matrix}\right." title="I(A) = \left\{\begin{matrix}1 &if&A&occurs & \\ 0 &if&A&doesn't&occur & \end{matrix}\right." /></a>

For example, flipping a fair coin, the sample space ```S = {H, T}``` and Probability of H = Probability of T = 1/2, we can define and indicator random variable X_H, associated with the coin coming up heads.

<a href="https://www.codecogs.com/eqnedit.php?latex=X_H&space;=&space;\left\{\begin{matrix}1&space;&if&H&occurs&space;&&space;\\&space;0&space;&if&T&occurs&space;&&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_H&space;=&space;\left\{\begin{matrix}1&space;&if&H&occurs&space;&&space;\\&space;0&space;&if&T&occurs&space;&&space;\end{matrix}\right." title="X_H = \left\{\begin{matrix}1 &if&H&occurs & \\ 0 &if&T&occurs & \end{matrix}\right." /></a>

The expected number of heads obtained in one flip

<a href="https://www.codecogs.com/eqnedit.php?latex=E[X_H]&space;=&space;1.P_r(H)&space;&plus;&space;0.P_r(T)&space;=&space;1/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[X_H]&space;=&space;1.P_r(H)&space;&plus;&space;0.P_r(T)&space;=&space;1/2" title="E[X_H] = 1.P_r(H) + 0.P_r(T) = 1/2" /></a>

The indicator random variable is good for repeated random trials. Let X_i be the indicator R.V. of i-th flip comes up head and X be the random variable denoting the total number of heads in the ```n``` coin flips

<a href="https://www.codecogs.com/eqnedit.php?latex=X&space;=&space;\sum_{i=1}^{n}X_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X&space;=&space;\sum_{i=1}^{n}X_i" title="X = \sum_{i=1}^{n}X_i" /></a>

Expectation of X is (and by applying **linearity of expectation**

<a href="https://www.codecogs.com/eqnedit.php?latex=E[X]&space;=&space;\sum_{i=1}^{n}E[X_i]&space;=&space;\sum_{i=1}^{n}&space;1/2&space;=&space;n/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[X]&space;=&space;\sum_{i=1}^{n}E[X_i]&space;=&space;\sum_{i=1}^{n}&space;1/2&space;=&space;n/2" title="E[X] = \sum_{i=1}^{n}E[X_i] = \sum_{i=1}^{n} 1/2 = n/2" /></a>

Returning to the hiring problem, the expected number of times we hire a new candidate, can be defined as ```n``` variables related to whether or not each particular candidate is hired. Let, X_i indicator R.V be the i-th candidate is hired and ```X = X1+ X2 + .. + Xn```.

Candidate i is hired exactly when candidate i is better than each of candidates 1 through (i - 1). Because we have assumed that the candidates arrive in a random order, the first i candidates have appeared in a random order. Any one of these first i candidates is equally likely to be the best-qualified so far. Candidate i has a probability of ```1/i``` of being better qualified than candidates 1 through (i - 1) and thus a probability of ```1/i``` of being hired.

<a href="https://www.codecogs.com/eqnedit.php?latex=E[X]&space;=&space;\sum_{i=1}^{n}E[X_i]&space;=&space;\sum_{i=1}^{n}&space;1/i&space;=&space;\ln{n}&space;&plus;&space;\Theta&space;(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[X]&space;=&space;\sum_{i=1}^{n}E[X_i]&space;=&space;\sum_{i=1}^{n}&space;1/i&space;=&space;\ln{n}&space;&plus;&space;\Theta&space;(1)" title="E[X] = \sum_{i=1}^{n}E[X_i] = \sum_{i=1}^{n} 1/i = \ln{n} + \Theta (1)" /></a>

Even though we interview n people, we actually hire only ```ln n``` of them on average.

#### Hat-Check Problem

**Q.** Each of n customers gives a hat to a hat-check person at a restaurant. The hat-check person gives the hats back to the customers in a random order. What is the expected number of customers who get back their own hat?

Let R_i be the indicator R.V of the i-th man gets his own hat back, so R_i = 1 is the event that he gets his own hat and R_i = 0 if he gets wrong hat. The number of men that get their own hat is the sum of these indicators:

```R = R1+ R2 + .. + Rn```

Now, each man gets his own hat with probability ```1/n```. So, using the linearity of expectation, we can have 

<a href="https://www.codecogs.com/eqnedit.php?latex=E[R]&space;=&space;\sum_{i=1}^{n}E[R_i]&space;=&space;\sum_{i=1}^{n}&space;1/n&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[R]&space;=&space;\sum_{i=1}^{n}E[R_i]&space;=&space;\sum_{i=1}^{n}&space;1/n&space;=&space;1" title="E[R] = \sum_{i=1}^{n}E[R_i] = \sum_{i=1}^{n} 1/n = 1" /></a>

So, we should expect 1 man to get his own hat back on average!

Ref:
- http://web.mit.edu/neboat/Public/6.042/expectationvalue1.pdf
- https://proofwiki.org/wiki/Hat-Check_Problem

### Revisiting Hiring Problem

To get the expected hiring cost, we introduce random behaviour what we discussed earlier.

#### Pseudocode
```
Random-Hiring-Problem(n)
1. randomly permute the list of candidates
2. best = 0 // dummy candidates
3. for i = 1 to n
4.     interview i
5.     if i > best
6.        best = i
7.        hire candidate i
```

With this simple change, we have created a randomized algorithm whose performance matches that obtained by assuming that the candidates were presented in a random order and the expected hiring cost is ```O(cost_of_hire * ln n)```. Now, the question is how to randonly permute the list of collection data? In next section, we will discuss the approach for randomly permutting array.

### Randomly permuting arrays

We assume that the collection (think array) contains the data 1 to n. Assign each element A[i] of the array a random priority P[i], and then sort the elements of A according to these priorities. For example, if our initial array is A = {1,2,3,4} and we choose random priorities P = {36,3,62,19}, we would produce an array B = {2,4,1,3}, since the second priority is the smallest, followed by the fourth, then the first, and finally the third.

#### Pseudocode
```
Permute-By-Sorting(A, n): //n = A.length
1. create new array P[1..n]
2. for i = 1 to n
3.     P[i] = RANDOM(1, n^3)
4. sort A using P as sort keys //in java use Collections.sort(List<T> list, Comparator<? super T> c)
```
Step 4 would take most of time of this algorithm as we know the lower bound of sorting of the collection is ```O(nlogn)```. After sorting if P[i] is the j-th smallest priority, then A[i] lies in position j of the output.

```java
import java.util.Random;
import java.util.*;

public static void permuteSort(int[] arr, int n) {
	Random rand = new Random();
	int[] prob = new int[n];
	for (int i = 0; i < n; i++)
		prob[i] = rand.nextInt(n*n*n); //generate random int in range 0 to n^3

	final class Wrapper implements Comparable<Wrapper> {
		final int sort_key;
		final int value;

		public Wrapper (int sort_key, int value) {
			this.sort_key = sort_key;
			this.value = value;
		}

		@Override
		public int compareTo(Wrapper o) {
			return Integer.compare(sort_key, o.sort_key);
		}
	}

	final List<Wrapper> temp = new ArrayList<Wrapper> ();
	for (int i = 0; i < n; i++)
		temp.add(new Wrapper(prob[i], arr[i]));

	Collections.sort(temp);

	for (Wrapper w : temp)
		System.out.print(w.value + " ");
}
```
Running time ```O(nlogn)``` and space ```O(n)```.

### Random permutation in place

A better method for generating a random permutation is to permute the given array in place. The procedure Randomize-In-Place does so in O(n) time. In it's i-th iteration, it chooses the element A[i] randomly from among elements A[i] through A[n]. Subsequent to the i th iteration, A[i] is never altered.

#### Pseudocode
```
Randomize-In-Place(A, n):
1. for i = 1 to n
2.     swap A[i] with A[RANDOM(i, n)]
```
Running time ```O(n)``` and space ```O(1)```.

### The Birthday Paradox
