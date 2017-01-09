/******************************************************************************************************
Title:			traffic_circle.c
Author: 		Emma Kimlin
Created on:		December 12th, 2016
Description: 	Simulate traffic in a traffic circle given the mean arrival times and the probability
		that a car entering at one entrance will exit at a given exit.  
Purpose:	Practice Open MPI programming and design. 
Usage: 		Usage: mpirun -np <number of processors> traffic <input-parameter-file> [max number time steps]
Build With: 	make
******************************************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Sweiss_Utilities.h" /* Written by Stewart Weiss, modified by Emma Kimlin */
#include <errno.h>
#include <limits.h>

#define ROOT 0 				/* Process ID 								*/
#define EPSILON .000005			/* Used during float comparison 					*/
#define WINDOW_SIZE 100			/* Number of iterations over which average queue size is determined.    */
#define AVG_QUEUE_DIFFERENCE 50     	/* If the difference between consecutive average queue sizes falls
 					below this number, traffic circle has reached steady state. 		*/

/* 
 * @entrance: the entrance where the car is entering the traffic circle
 * Pre-condition: A car has arrived at the traffic circle and is ready to enter. 
 * Post-condition: A car has entered the traffic circle at this entrance and the exit
 * 		toward which this car is headed is decided. The exit is stored in 
 *		transition probabilities. 
*/
int carEntersTrafficCircle(int entrance,  double transition_probabilities[16], int* errval);
/*
 * @filename refers to a text file containing a matrix with 4 columns and 5 rows. The first row 
 *		holds mean arrival times for the North, East South and West entrances, respectively. 
 *		The remaining 4 rows hold the probabilities that a car that enters at one entrance 
 *		will exit at another. Each row sums to 1.
 * Post-condition: Transition_probabilities holds the probabilites linearly such that indices 0-3 hold 
 * 		the values in the 2nd row, 4-7 hold that values in the 3rd row, and so on. The first
 *		row is stored in mean_arrivals. 
*/
void read_input_matrix (char *filename, double transition_probabilities[16], double mean_arrivals[4], 
                                                                                        int* errval);
/*
 * Compares 2 floats to check if they are equal. If equal, returns 1. If not, returns 0. 
*/
int checkIfFloatsEqual(double a, double b);
/* 
 * Post-condition: fills arrival_time_exp_param[] with the respective inverses of mean_arrival_times[]
*/
void generateExponentialParameters(double mean_arrival_times[4], double arrival_time_exp_param[4], int* error);
/*
 * @queue_accum holds the sum of all queues over each iteration during the current window. 
 * @time_step_count is the total number of iterations or time_steps since start
 * @avg_queue_size holds the previous window's average queue sizes, if this is not the fist window. 
 * Post-condition: If this is the first window, avg_queue_size is populated with the average queue sizes 
 *		for this window and the function returns 0. 
 *		If this is not the first window, the average queue sizes of this window are calculated and 
 *		compared to the last window to see what the difference is between them. If the difference is 
 *		not greater than AVG_QUEUE_DIFFERENCE, the function returns 1. Otherwise returns 0. Either 
 * 		way, queue_accum is reset to 0.
 * 		If this was called before window is complete, return 0.		
*/																					
int checkIfReachedSteadyState(double avg_queue_size[4], int queue_accum_in_window[4], int time_step_count, 
															int* error, int reached_steady_state_bool[4]);

int main(int argc, char* argv[]) {
	int id; 			 /* Rank of executing process 						 */
	int p; 				 /* Number of processes 						 */
	int traffic_circle1[16];		 
	int traffic_circle2[16]; 							 
	int* traffic_circle_p = traffic_circle1; 
					 /* Each element represents segment of traffic circle that is either
					    empty or holds one car. If traffic_circle[i] = -1, segment is empty.
					    Otherwise circle[i] containts integer that represents the car's exit */
	int* advanced_traffic_circle_p = traffic_circle2;
					 /* Stores the state of the traffic circle after all cars in the circle 	 
					    have advanced in the current time step. 			 	 */
	int arrival[4] = {0}; 		 /* Boolean array. arrival[i] = 1 when a car arrives at an entrance 	 */
	double time_til_next_arrival[4]; /* Stores the time until the next car arrives at each entrance. 	 */
	int arrival_count[4] = {0};	 /* Contains total number of arrivals at each entrance across all 
					    processes								 */
	int local_arrival_count[4] = {0};/* Contains total number of arrivals at each entrance for this process  */
	int wait_count[4] = {0}; 	 /* Number of cars that could not enter circle immediately for each 
					    entrance over all processes						 */
	int local_wait_count[4] = {0}; 	 /* Number of cars that could not enter circle immediately for each 
					    entrance for this process. 						 */
	int queue[4] = {0};		 /* How many cars are waiting to enter traffic circle at each entrance
					    during this time step.					     	 */
	int queue_accum[4] = {0}; 	 /* Runnning total of values in queue over all time steps for all 
					    processes								 */
	int local_queue_accum[4] =  {0}; /* Running total of values in queue over all time steps for this 
					    process 								 */
	int queue_accum_in_window[4] = {0}; /* Running total of values in queue within current window for all
					       processes							 */
	int local_queue_accum_in_window[4] = {0}; /* Running total of values in queue within current window for
						this process.							 */
	double avg_queue_size[4] = {0};  /* Stores the average queue size at each entrance over a window of 
					    WINDOW_SIZE iterations						 */
	double mean_arrival_time[4];     /* The probability of a car arriving at an entrance during a particular 
					    time step is a random variable from an exponential distribution with 
					    mean m. This array holds the mean time between arrival at each 
					    entrance. 								 */
	double arrival_time_exponential_parameter[4]; 
					/* Holds the exponential parameter, calculated as the inverse of the
					   mean arrival time, for each entrance. To be used when deciding if
					   a car has arrived or not.						 */
	double transition_probabilities[16];  
					/* Store probabilities that a car entering at entrance i will
					   exit at exit j. 					 		 */
	int reached_steady_state[4]; 	/* For each entrance, indicates whether entrance has reached steady 
					   state. Must all be set to 1 in order for reached_steady_state_bool 
					   to be true			 					 */
	int reached_steady_state_bool = 0;/* Boolean to indicate whether or not the traffic circle has reached 
					a steady state (defined by the changed in the average queue size) 	 */
	int time_step_count = 0; 	/* Stores total number of iterations of the while loop across all 
					   processes 								 */
	int local_time_step_count = 0;
	int error = 0; 
	double random_num;
	int* swap_helper;
	int max_time_steps = 100000;    /* Maximum number of iterations the while loop runs for. Default to 100k
					   and will update if user provided 2nd command line argument 		 */

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
/* CHECK AND DISTRIBUTE USER INPUT */
    if (id == p-1) {
        if ((argc < 2) || (argc > 3)) { //Check number of arguments user provided
            printf("Usage: traffic_circle <input-parameter-file> [max number time steps] \n");
            error = 1;
        } else 
            read_input_matrix (argv[1], transition_probabilities,  
                                       mean_arrival_time, &error); //Read matrix from file 
        if (error != 1)
        	generateExponentialParameters(mean_arrival_time, arrival_time_exponential_parameter, &error); 
        if (error != 1 && argc == 3) {
        	char* err;
        	max_time_steps = strtol(argv[2], &err, 10);
        	if (*err != '\0' || max_time_steps > INT_MAX || max_time_steps < 1) {
        		printf("[max time steps] must be an integer greater than 0. \n");
        		error = 1;
        	}
        }
    }
    MPI_Bcast (&error, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	if (error == 1) {
        MPI_Finalize();
        return 1;
    } 
    MPI_Bcast(mean_arrival_time, 4, MPI_DOUBLE, p-1, MPI_COMM_WORLD);
    MPI_Bcast(transition_probabilities, 16, MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(arrival_time_exponential_parameter, 4, MPI_DOUBLE, p-1, MPI_COMM_WORLD);
	MPI_Bcast(&max_time_steps, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	/* FIND OUT WHEN THE FIRST CARS ARRIVE FOR EACH ENTRANCE */
	for (int i = 0; i < 4; ++i) {
		random_num = (double) (random()) / RAND_MAX; 
    	time_til_next_arrival[i] = ( - log (random_num) / arrival_time_exponential_parameter[i]);
	}
	/* INITIALIZE TRAFFIC CIRCLES TO -1 BECAUSE THERE ARE NO CARS IN THEM */
	for (int i = 0; i < 16; ++i) {
		traffic_circle1[i] = -1;
		traffic_circle2[i] = -1;
	}

	/* BEGIN TIME SIMULATION OF THE TRAFFIC CIRCLE */
	while ((reached_steady_state_bool == 0) && (time_step_count < max_time_steps)) {
		/* STEP 1: SEE IF NEW CARS HAVE ARRIVED AT TRAFFIC CIRCLE ENTRANCES */
		for (int i = 0; i < 4; ++i) {
			/* Determine if car has arrived or not. If one has arrived, determine the time until
			   the next car arrives.  */
			if (time_til_next_arrival[i] < 1) {
				++local_arrival_count[i];
				arrival[i] = 1; //Indicate that a new car has arrived in arrivals[]
				random_num = (double) (random()) / RAND_MAX; //Find out when the next car will arrive
    			time_til_next_arrival[i] += ( - log (random_num) / arrival_time_exponential_parameter[i]);
			}
		}

		/* STEP 2: AT EACH SEGMENT, MOVE CARS FORWARD OR EXIT */
		for (int i = 0; i < 16; ++i) {
			/* If a car is one space before the exit that is its destination exit, remove the car from
				 advanced_traffic_circle in the next time segment.   */
			if ( ((i == 14) && (*(traffic_circle_p + i) == 0)) || (( i == 2) && (*(traffic_circle_p + i) == 1)) ||
				((i == 6) && (*(traffic_circle_p + i) == 2)) || 
			    	((i == 10) && (*(traffic_circle_p + i) == 3)) ) {
				*(advanced_traffic_circle_p + i + 1) = -1;
			}
			/* If a car is not one before an exit, move it forward 1 segment  */
			else {
				if (i == 15) //This car moves to the beginning of the traffic circle
					*(advanced_traffic_circle_p) = *(traffic_circle_p + i);
				else 
					*(advanced_traffic_circle_p + i + 1) = *(traffic_circle_p + i);
			}
		} //End Step 2

		/* STEP 3: ENTER TRAFFIC CIRCLE IF POSSIBLE. OTHERWISE, CAR MUST WAIT IN QUEUE */
		/* At each entrance: */
		for (int i = 0; i < 4; ++i) {
			/* Case 1: If no new car has arrived, but there are cars waiting to enter and the 
			   entrance segment will be free in the next time segment, let a car enter from the queue. */
			if ((arrival[i] == 0) && (queue[i] > 0) && (*(advanced_traffic_circle_p + i*4) == -1)) {
				--queue[i]; 
				*(advanced_traffic_circle_p + i*4) = carEntersTrafficCircle(i, transition_probabilities, &error);
                if (error == 1) return 1;
			}
			/* Case 2: A car has arrived, the queue is empty and the entrance segment will be free 
			   in the next time segment, the car that just arrived enters traffic circle immediately  */
			else if ((arrival[i] == 1) && (queue[i] == 0) && (*(advanced_traffic_circle_p + i*4) == -1)) {
				*(advanced_traffic_circle_p + i*4) = carEntersTrafficCircle(i, transition_probabilities, &error);
                if (error == 1) return 1;
			}
			/* Case 3: If a car has arrived and the entrance segment will be taken 
				in the next time segment, the car that just arrived must wait and is put into queue */
			else if ((arrival[i] == 1) && (*(advanced_traffic_circle_p + i*4) != -1)) {
				++queue[i];
				++local_wait_count[i];
			}
			/* Case 4: Car has arrived, queue is not empty and entrance segment will be free in the 
						next time segment, so recently arrived car is added to queue and a car that 
						has been waiting enters the traffic circle */
			else if ((arrival[i] == 1) && (queue[i] > 0) && (*(advanced_traffic_circle_p + i*4) == -1)) {
				*(advanced_traffic_circle_p + i*4) = carEntersTrafficCircle(i, transition_probabilities, &error);
                if (error == 1) return 1;
				++local_wait_count[i];
			}
			/* In all other cases, no car has arrived and entrance segment will
				not be free in the next time segment. Do nothing. */
			/* Reset necessary values for next time step: */ 
			local_queue_accum[i] += queue[i];
			local_queue_accum_in_window[i] += queue[i];
			arrival[i] = 0;
			time_til_next_arrival[i] = time_til_next_arrival[i] < 0 ? 0 : --time_til_next_arrival[i];
		} //End Step 3
		/* Reset necessary values for next time step: */ 
		swap_helper = traffic_circle_p;
		traffic_circle_p = advanced_traffic_circle_p; //Traffic circle now holds all values that were in Advanced Traffic Circle
		advanced_traffic_circle_p = swap_helper;  
		++local_time_step_count;
		/* Calculate the total number of iterations of all processes. The only values that need to be processed 
		   are the ones required to determine when to exit while loop */
		MPI_Reduce(local_queue_accum_in_window, queue_accum_in_window, 4, MPI_INT, MPI_SUM, p-1, MPI_COMM_WORLD);
		MPI_Bcast(queue_accum_in_window, 4, MPI_INT, p-1, MPI_COMM_WORLD);
		MPI_Reduce(&local_time_step_count, &time_step_count, 1, MPI_INT, MPI_SUM, p-1, MPI_COMM_WORLD);
		MPI_Bcast(&time_step_count, 1, MPI_INT, p-1, MPI_COMM_WORLD);
		/* If on the edge of a window, process p-1 updates the average queue sizes for each entrance and checks if 
		   each entrance has a reached steady state.  */
		if (id == p-1) {
			if (time_step_count % WINDOW_SIZE == 0) {
				reached_steady_state_bool = checkIfReachedSteadyState(avg_queue_size, queue_accum_in_window, time_step_count,
																							&error, reached_steady_state);
				if (error == 1) {
					printf("Error calculating if program reached steady state. \n");
					return 1;
				}
			} 
		}
		MPI_Bcast(&reached_steady_state_bool, 1, MPI_INT, p-1, MPI_COMM_WORLD);
	} //End While
	/* OUTPUT RESULTS: Find average queue size across all iterations. Output average length of
		the queue across all iterations and the probability that a car will have to wait before 
		entering the circle.  */
	MPI_Reduce(local_queue_accum, queue_accum, 4, MPI_INT, MPI_SUM, p-1, MPI_COMM_WORLD);
	MPI_Reduce(local_arrival_count, arrival_count, 4, MPI_INT, MPI_SUM, p-1, MPI_COMM_WORLD);
	MPI_Reduce(local_wait_count, wait_count, 4, MPI_INT, MPI_SUM, p-1, MPI_COMM_WORLD);
	if (id == p-1) {
		printf("Results: \n Average Queue size across all iterations: \n");
	 	for (int i = 0; i < 4; ++i) {
	 		avg_queue_size[i] = queue_accum[i] / time_step_count; //Calculate average queue size across all iterations
	 		printf("%.2f \n", avg_queue_size[i]);
	 	}
	 	printf("Probability that a car will have to wait: \n");
	 	for (int i = 0; i < 4; ++i) {
	 		printf("Entrance %i : %.2f %% \n", i, 100* ((double)wait_count[i]/arrival_count[i]));
	 	}
	}
    MPI_Finalize();
    return 0;
}
 
int carEntersTrafficCircle(int entrance, double transition_probabilities[16], int *errval) {
	/* Find out what exit this car is going to use */
	int destination = -1;
	double random_var = uniform_random();
	for (int possible_destination = 0; possible_destination < 4; ++possible_destination) {
		if (random_var <= transition_probabilities[possible_destination + (4*entrance)]) {
			destination = possible_destination; //destination will be 0, 1, 2 or 3 
			break;
		}
	}
	if (destination == -1) {
		*errval = 1; 
		printf("Error assigning destination to car entering circle. \n");
		return -1;
	} else 
		return destination; 
}

void read_input_matrix (char *filename, double transition_probabilities[16], 
          								double mean_arrivals[4], int *errval)
{   
    FILE   *file;           /* input file stream pointer */
    double input_data[20];  /* Holds the 20 values from the input file       */
    double sum_of_line;     /* Holds the sum of a line in the input file     */
    int n = 0;
    int read_count = 0;     /* Number of values read from input file         */
    /* Process p-1 opens the binary file containing the matrix, 
        inputs it into a linear array, checks for errors and 
        checks values are valid. */
    file = fopen (filename, "r");
    if ( NULL == file ) {
        *errval = 1;
        return;
    }
    /* Read each value in each line into input_data[] */
    for (int i = 0; i < 20; ++i) {
        if (fscanf(file,"%lf %lf %lf %lf", &input_data[n], &input_data[n+1], &input_data[n+2],
        																&input_data[n+3]) == 4) {
            n+=4;
            read_count +=4;
        }
    }
    if (read_count != 20) {
        *errval = 1;
        return;
    }
    /* Store mean arrival times (first 4 values in input_data) in mean_arrivals[] */
    for (int i = 0; i <= 4; ++i)
        mean_arrivals[i] = input_data[i];
    /* Check that the last 4 lines of the input file, which hold transition probabilities
        for each entrance, each total to 1. */
    for (int n = 4; n < 20; n+=4) {
        /* Make sure row adds to 1 */
        sum_of_line = input_data[n] + input_data[n+1] + input_data[n+2] + input_data[n+3];
        if ( checkIfFloatsEqual(1.0, sum_of_line) == 0) {
            *errval = 1;
            return;
        } else { /* If it does add to 1, move into transition_probabilities[] */
            transition_probabilities[n-4] = input_data[n];  
            transition_probabilities[n-3] = transition_probabilities[n-4] + input_data[n+1]; 
            transition_probabilities[n-2] = transition_probabilities[n-3] + input_data[n+2];
            transition_probabilities[n-1] = transition_probabilities[n-2] + input_data[n+3]; 
        }    
    }
}

int checkIfFloatsEqual(double a, double b) {
    return fabs(a - b) < EPSILON;
}

void generateExponentialParameters(double mean_arrival_times[4], double arrival_time_exp_param[4], 
																						int* error) 
{
	for (int i = 0; i < 4; ++i) {
		/* Invert mean to get exponential parameter, if possible */
    	if ( 0 != mean_arrival_times[i] )
        	arrival_time_exp_param[i] = 1.0 / mean_arrival_times[i]; 
    	else {
        	*error = 1;
        	return;
        }
	}
}

int checkIfReachedSteadyState(double avg_queue_size[4], int queue_accum_in_window[4], int time_step_count, int* error,
								int reached_steady_state[4]) {
	double avg; 
	int reached_steady_state_bool = 0;
	for (int i = 0; i < 4; ++i) {
		/* If this is the first window, calculate average queue size and store in avg_queue_size[] */
		if (time_step_count == WINDOW_SIZE) {
			avg_queue_size[i] = queue_accum_in_window[i] / WINDOW_SIZE; 
			queue_accum_in_window[i] = 0; //Reset accumulation of queues for new window
		} else if (time_step_count % WINDOW_SIZE == 0) {
		/* If this is not the first window, calculate current window's average queue size and 
		compare to last windows average queue size (which is stored in avg_queue_size[])	 */
			avg = queue_accum_in_window[i] / WINDOW_SIZE; //current window's average queue 
			if (fabs(avg_queue_size[i] - avg) <= AVG_QUEUE_DIFFERENCE) //reached steady state for this entrance
				reached_steady_state[i] = 1; /* The difference between this avg qeueue size and the last avg queue size is */
			else 						  /* below the threshold. therefore this entrance is in steady state 			*/
				reached_steady_state[i] = 0;
			avg_queue_size[i] = avg; /* Replace last average with current average */
			queue_accum_in_window[i] = 0; //Reset accumulation of queues for new window
		} else {
		/* This function should not have been called; program is mid-window. */
			*error = 1;
		}
	}
	if ((reached_steady_state[0] == 1) && (reached_steady_state[1] == 1) && (reached_steady_state[2] == 1) 
																		&& (reached_steady_state[3] == 1))
		reached_steady_state_bool = 1;
	return reached_steady_state_bool;
}










