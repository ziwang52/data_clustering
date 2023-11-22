//Zi Wang
//phase 5
// c++ coding standards http://web.mit.edu/6.s096/www/standards.html
/*
line 309 --- external function call
line 427 ----external function detail
*/

#include <iostream>
#include <fstream>
#include <string>
#include<random>
#include<cfloat>

using namespace std;

void cluster(double** datastore, double** cluster_centers, int* labels, int num_centers, int num_points, int num_attributes);
double getdistance(double datapoints[], double centers[], int num_attributes);
double getSSE(int* labels, double** datapoints, double** centers, int num_centers, int num_points, int num_attributes);
void singleton_cluster(int* labels, double** datapoints, double** centers, int num_centers, int num_points, int num_attributes);
void getcenters(int* labels, double** datastore, double** new_centers, int num_centers, int num_points, int num_attributes);
//normalization
void min_max_normalization(double** temp, double** datastore, double** value, double* max, double* min, double num_points, int num_attributes);
void z_score(double** temp, double** datastore, double** z_value, double* mean, double num_points, int num_attributes);
//initialization
void random_partition(int* labels, double** datastore, double** partition_centers, int num_centers, int num_points, int num_attributes);
void random_select(double** temp, double** clusters, int num_centers, int num_points, int num_attributes);
void maximin(double** temp, double** clusters, int num_centers, int num_points, int num_attributes);

//internal_Validation function
double calinski_Harabasz(double** datastore, double S_w, int num_centers, int num_points, int num_attributes);
double silhouette(double** datastore, int* labels, double** clusters, int num_centers, int num_points, int num_attributes);
double davies_Bouldin(double** datastore, int* labels, double** clusters, int num_centers, int num_points, int num_attributes);
double dunn(double** datastore, int* labels, double** clusters, int num_centers, int num_points, int num_attributes);

// external Validation function
void external_index(double& rand, double& jaccard, double& fowlkes, int num_points, int* labels, int* true_labels);



int main(int argc, char* argv[])//int argc, char* argv[]
{ 
  //Variable Declaration/Initialization
  string filename = argv[1];
  int num_centers = atoi(argv[2]);
  int num_iteration = atoi(argv[3]);
  double  threshold = atoi(argv[4]);
  int num_run = atoi(argv[5]);
 
  int i, num_points, num_attributes;
  double S_within = 0.0;

  /*
  string filename = "iris_bezdek.txt";
  int num_centers = 3;
  int num_iteration = 100;
  double  threshold = 0.001;
  int num_run = 100;
*/
 //Open file for reading
  fstream in_file(filename, ios::in);

  //Test for open
  if (!in_file)
  {
    cout << "Cannot open " << filename << " for input" << endl;
    return 0;
  }

  //read and store number of points(points) and the dimensionality of each point (D)
  for (i = 0; i < 1; i++)
  {
    in_file >> num_points >> num_attributes >> num_centers;
  }

  //cout<<"file name: "<<filename<<" num points: "<<num_points << " num_attributes: " <<num_attributes << " num_centers: " <<num_centers<<endl;
 // create array for true label
  int* true_label = new int[num_points];
  //**temp to read datasets
  double** temp;
  temp = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    temp[i] = new double[num_attributes];
  }

  // read data into 2d array
  for (i = 0; i < num_points; i++)
  {
    for (int j = 0; j < num_attributes; j++)
    {
      in_file >> temp[i][j];
    }
  }

//store true label
  for (i = 0; i < num_points; i++)
  {
    for (int j = num_attributes - 1; j < num_attributes; j++)
    {
     true_label[i]= temp[i][j]  ;
    }

  }


//get num attributes -1;
num_attributes=num_attributes-1;

  double** datastore = NULL;
  datastore = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    datastore[i] = new double[num_attributes];
  }

//store raw data
  double** temp_data = NULL;
  temp_data = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    temp_data[i] = new double[num_attributes];
  }
for (i = 0; i < num_points; i++)
  {
    for (int j = 0; j < num_attributes ; j++)
    {
      datastore[i][j] = temp[i][j];
      temp_data[i][j] = temp[i][j];
    }
  }


 


  //define k max value
  double K_max = ceil(sqrt(num_points / 2));
  //cout<<"k max:"<<K_max<<endl;


  // Dynamically allocate 2D array  with N/D
 
  // **clusters to store first cluster random centers

  // **datastore to store dataset 
  
  //partition_centers to use random partition method
  double** partition_centers;
  partition_centers = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    partition_centers[i] = new double[num_attributes];
  }
  //delcare for min-max normaliztion
  double** value = NULL;
  value = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    value[i] = new double[num_attributes];
  }
  //declare z-value for z score
  double** z_value = NULL;
  z_value = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    z_value[i] = new double[num_attributes];
  }
  double* max = new double[num_attributes];
  double* min = new double[num_attributes];
  double* mean = new double[num_attributes];




  //declare array labels to store datapoints  for lableing  datapoints depend on different cluster centers 
  int* labels = NULL;
  labels = new int[num_points];

  //declare array result to find min run of SSE 
  double* result = NULL;
  result = new double[num_run + 1];

  double* inital_SSE = NULL;
  inital_SSE = new double[num_run + 1];

  int* iter_run = NULL;
  iter_run = new int[num_run + 1];

  //write output into TXT
  string outfile = filename + ".out.txt";
  fstream out_file(outfile, ios::out);

  //Test for open
  if (!out_file)
  {
    cout << "Cannot open " << outfile << " for output" << endl;
  }
  else
  {   // random select from input files
    cout << "\n\n writing output into TXT file..." << endl << endl;
  }


  double best_run = DBL_MAX;
 

    double** clusters;
    clusters = new double* [num_centers];
    for (i = 0; i < num_centers; i++)
    {
      clusters[i] = new double[num_attributes];
    }

// create array to find best external index
    double* rand_index = NULL;
    rand_index = new double[num_run + 1];

    double* jaccard_index = NULL;
    jaccard_index = new double[num_run + 1];

    double* fowlkes_index = NULL;
    fowlkes_index = new double[num_run + 1];
 


    // using for loop to run R(100) times
    for (int ii = 1; ii <= num_run; ii++)
    {
      
        double rand   =0.0;
       double jaccard = 0.0;
        double fowlkes = 0.0;
      


      iter_run[ii] = 0;
      //delcare variables curr_SSE/2 for SSE results
      int count = 0;
      double  curr_SSE = 0.0;
      double  prev_SSE = 0.0;
     //cout << "\nRUN  " << ii << endl;
      out_file << "\nRUN  " << ii << endl;
     // cout << "------" << endl;
      out_file << "------" << endl;

      //NORMALIZATION 

        // z_score(temp, datastore, z_value, mean, num_points, num_attributes);
      min_max_normalization(temp_data, datastore, value, max, min, num_points, num_attributes);

      // (**value is min_max_normalization, **z_value is z_score)

        //Initialization

    //  random_partition(labels,  datastore, clusters,   num_centers,   num_points,  num_attributes);
      random_select(datastore,  clusters,  num_centers,  num_points,   num_attributes);
     // maximin(datastore, clusters, num_centers, num_points, num_attributes);

 
 //first run cluster base on random centers 
      cluster(datastore, clusters, labels, num_centers, num_points, num_attributes);
      // get first SSE result base on first run 
      curr_SSE = getSSE(labels, datastore, clusters, num_centers, num_points, num_attributes);
      inital_SSE[ii] = curr_SSE;

      count++;
 
 // cout << "iteration 1 SSE: " << curr_SSE << endl;
      out_file << "iteration 1 SSE: " << curr_SSE << endl;
      // get new cluster centers     
      getcenters(labels, datastore, clusters, num_centers, num_points, num_attributes);


      //run cluster again base on new centers
      cluster(datastore, clusters, labels, num_centers, num_points, num_attributes);
      prev_SSE = getSSE(labels, datastore, clusters, num_centers, num_points, num_attributes);

      //singleton_clusters check  
//singleton_cluster(labels, datastore, clusters, num_centers, num_points, num_attributes);

      count++;
      out_file << "iteration 2 SSE: " << prev_SSE << endl;
      // repeat previous steps until meet (T<0.001), using fabs() to return a absolute value for the accurate result    
      while ((fabs((prev_SSE - curr_SSE) / prev_SSE)) > threshold)
      {
        curr_SSE = prev_SSE;
        getcenters(labels, datastore, clusters, num_centers, num_points, num_attributes);
        cluster(datastore, clusters, labels, num_centers, num_points, num_attributes);
        prev_SSE = getSSE(labels, datastore, clusters, num_centers, num_points, num_attributes);
        // singleton_cluster(labels, datastore, clusters, num_centers, num_points, num_attributes);

        count++;
        iter_run[ii] = count;
        // cout << "iteration " << count << " SSE: " <<  prev_SSE  << endl;

        out_file << "iteration " << count << " SSE: " << prev_SSE << endl;
        //stop when match max iteration num 
        if (count > num_iteration)
        {
          //cout << "max iteration num!!" << endl;
          break;
        }
      }
      //using result to record the final SSE of Each run
      result[ii] = prev_SSE;
      S_within = prev_SSE;
//external calculation
      external_index( rand,  jaccard,  fowlkes,  num_points,  labels, true_label);

     // cout<<"run "<<ii<<" rand:  "<< rand<<"  jaccard:  "<< jaccard <<"  fowlkes: "<< fowlkes<<endl;
      // internal_Validation with best run in 100 
      if (best_run > result[ii])
      {
        best_run = result[ii];
     

      }
      rand_index[ii] = rand;
      jaccard_index[ii] = jaccard;
      fowlkes_index[ii] = fowlkes;
    }                     


    //find the best run of SSE with its index which the is RUN num
    double best = DBL_MAX;
    double best_inital_SSE = DBL_MAX;
    int min_iter = 100;
    int initial_SSE_index = 1;

    int run = 0;

    double best_rand=0.0;
    double best_jaccard =0.0;
    double best_fowlkes=0.0;
     double  best_rand_index     =0.0;
     double  best_jaccard_index  =0.0;
     double  best_fowlkes_index  =0.0;

    for (int index = 1; index <= num_run; index++)
    {
     // find  best value of external validity index  
      if (rand_index[index] > best_rand)
      {
        best_rand= rand_index[index];
        best_rand_index= index;
      }

      if (jaccard_index[index] > best_jaccard)
      {
        best_jaccard = jaccard_index[index];
        best_jaccard_index = index;
      }
      if (fowlkes_index[index] > best_fowlkes)
      {
        best_fowlkes = fowlkes_index[index];
        best_fowlkes_index = index;
      }

      
      //best iteration num
      if (min_iter > iter_run[index])
      {
        min_iter = iter_run[index];
      }
      //best inital SSE
      if (best_inital_SSE > inital_SSE[index])
      {
        best_inital_SSE = inital_SSE[index];
        initial_SSE_index = index;
      }
      //best final SSE
      if (best > result[index])
      {
        best = result[index];
        run = index;
      }

    }
cout<<"R = 100, Normalization: min_max,   Initialization: random selected"<<endl;
  cout<<"\n\nBest Run:  "<<best_rand_index<<"  Best Rand value: "<<best_rand<<endl;
  cout << "Best Run:  " << best_jaccard_index << "  Best Jaccard value: " << best_jaccard << endl;
  cout << "Best Run:  " << best_fowlkes_index << "  Best  Fowlkes-Mallows value: " << best_fowlkes << endl;

    //cout<<"best Run : "<< initial_SSE_index<<"  inital_SSE: "<< best_inital_SSE<<endl;
    cout << "Best Run: " << run << " : SSE = " << best << endl;
    //cout<<"\nBest Number of iterations: "<<min_iter<<endl;
    //  out_file << "\n\nBest Run: " << run << " : SSE = " << best << endl;
    for (i = 0; i < num_centers; i++)
    {
      delete[] clusters[i];
    }
    delete[] clusters;
  

  //close the files and clear memory
  in_file.close();
  out_file.close();


  for (i = 0; i < num_points; i++)
  {
    delete[] temp_data[i];
    delete[] temp[i];
    delete[] datastore[i];
    delete[] partition_centers[i];
    delete[] value[i];
    delete[] z_value[i];
  }
  delete[] temp_data;
  delete[]true_label;
 delete [] jaccard_index;
 delete[] fowlkes_index;
 delete[]rand_index;
  delete[] temp;
  delete[] datastore;
  delete[] labels;
  delete[] result;
  delete[] partition_centers;
  delete[] min;
  delete[] max;
  delete[] value;
  delete[] z_value;
  delete[]mean;
  delete[]inital_SSE;
  delete[]iter_run;
  return 0;
}
void external_index(double& rand, double& jaccard, double &fowlkes, int num_points, int* labels, int* true_labels )
{
  int i, j;
  double total =0.0;
  double true_pos = 0;
  double true_neg = 0;
  double false_pos = 0;
  double false_neg = 0;
   // get external index
  for (i = 0; i < num_points; i++)
  {
    for (j = i + 1; j < num_points; j++)
    {
      //get TP  
      if ((true_labels[i] == true_labels[j]) && (labels[i] == labels[j]))
      {
        true_pos+=1;
      }
      // GET FN
      if ((true_labels[i] == true_labels[j]) && (labels[i] != labels[j]))
      {
        false_neg += 1;
      }
      //GET FP
      if ((true_labels[i] != true_labels[j]) && (labels[i] == labels[j]))
      {
        false_pos += 1;
      }
      //GET TN
      if ((true_labels[i] != true_labels[j]) && (labels[i] != labels[j]))
      {
        true_neg += 1;
      }

    }
  }
  //GET TOTAL
  total = (true_pos + false_neg + false_pos + true_neg);



//compute rand
  rand = ((true_pos + true_neg) / total);
//compute fowlkes
  fowlkes = true_pos / (sqrt((true_pos + false_neg) * (true_pos + false_pos)));
  // compute jaccard

  jaccard = (true_pos / ( true_pos + false_neg +false_pos));

  
 }

double dunn(double** datastore, int* labels, double** clusters, int num_centers, int num_points, int num_attributes)
{
  int i, j;
  double sum = 0.0;
  double max_dis = 0.0;
  double min_dis = 0.0;
  double max_intra = 0.0;
  double min_inter = DBL_MAX;
  double dunn = 0.0;
  // get max intra distance

  for (i = 0; i < num_centers; i++)
  {
    for (j = 0; j < num_points; j++)
    {
      if (i == labels[j])
      {
        // cout<<"labels; "<<i<<endl;
        sum = getdistance(datastore[j], clusters[i], num_attributes);
        if (sum > max_intra)
        {
          max_intra = sum;
        }
      }


    }



  }
  //get min inter 
  for (i = 0; i < num_centers; i++)
  {
    for (j = 0; j < num_points; j++)
    {
      if (i == labels[j])
      {

        for (int k = 0; k < num_points; k++)
        {
          if (i != labels[k])
          {
            min_dis = getdistance(datastore[j], datastore[k], num_attributes);
            if (min_dis < min_inter)
            {
              min_inter = min_dis;
            }
          }
        }
      }
    }
  }





  max_intra = 2 * max_intra;
  //double  distance to get diameter distance 


  //cout << "min: " << min_inter << "  max: " << max_intra << endl;
  dunn = min_inter / 2 * max_intra;
  //cout << " dunn index: " << dunn << endl;
  return dunn;
}

double davies_Bouldin(double** datastore, int* labels, double** clusters, int num_centers, int num_points, int num_attributes)
{
  int i, j;
  double sum = 0.0;
  double* average_dis = new double[num_centers];

  // get average dis whithin clusters

  for (i = 0; i < num_centers; i++)
  {
    average_dis[i] = 0;
    int count = 0;
    for (j = 0; j < num_points; j++)
    {

      if (i == labels[j])
      {
        // cout<<"labels; "<<i<<endl;
        average_dis[i] += getdistance(datastore[j], clusters[i], num_attributes);
        count++;
      }
    }
    average_dis[i] = average_dis[i] / count;
    //cout << "average " << average_dis[i] << endl;
  }

  double total = 0.0;
  double max = 0.0;
  double dbi = 0.0;

  //get max distance betweem clusters
  for (i = 0; i < num_centers; i++)
  {
    for (j = i + 1; j < num_centers; j++)
    {
      total = average_dis[i] + average_dis[j];
      total = total / getdistance(clusters[i], clusters[j], num_attributes);

      if (total > max)
      {
        max = total;
      }
    }
    dbi += max;
  }
  dbi = dbi / num_centers;

  //cout << " DBI:  " << dbi << endl;


  delete[] average_dis;
  return dbi;
}

double silhouette(double** datastore, int* labels, double** clusters, int num_centers, int num_points, int num_attributes)
{

  int i, j;
  double* a_dis = new double[num_points];
  double* b_dis = new double[num_points];
  double* temp = new double[num_points];
  double min = DBL_MAX;
  double silhouette_score = 0.0;
  int sum = 0;
  for (j = 0; j < num_points; j++)
  {
    a_dis[j] = 0;
    b_dis[j] = 0;
    temp[j] = 0;
  }

  for (j = 0; j < num_points; j++)
  {
    // get  distance within   cluster

    a_dis[j] = getdistance(datastore[j], clusters[labels[j]], num_attributes);
    // cout<<"a_dis: "<< a_dis[j]<<endl;


    int count = 0;
    // get min out cluster distance
    for (i = 0; i < num_centers; i++)
    {
      if (labels[j] != i)
      {
        count++;
        temp[count] = getdistance(datastore[j], clusters[i], num_attributes);
        if (min > temp[count])
        {

          min = temp[count];

        }
      }

    }

    b_dis[j] = min;
    // cout << "b_dis: " << b_dis[j] << endl;

  }

  // sum all get final sw score max(a,b)
  for (i = 0; i < num_points; i++)
  {
    if (a_dis[i] > b_dis[i])
    {
      silhouette_score += ((b_dis[i] - a_dis[i]) / a_dis[i]);
    }
    if (a_dis[i] < b_dis[i])
    {
      silhouette_score += ((b_dis[i] - a_dis[i]) / b_dis[i]);
    }
  }
  silhouette_score = silhouette_score / num_points;
  //cout << "silhouette_score:  " << silhouette_score << endl;


  delete[] a_dis;
  delete[] b_dis;
  delete[] temp;
  return silhouette_score;
}
double calinski_Harabasz(double** datastore, double S_w, int num_centers, int num_points, int num_attributes)
{
  int i, j;
  double Calinski_Harabasz_score = 0.0;
  double S_total = 0.0;
  double S_between = 0.0;
  double* sum = NULL;
  sum = new double[num_attributes];
  double* global_center = NULL;
  global_center = new double[num_attributes];

  for (int k = 0; k < num_attributes; k++)
  {
    sum[k] = 0.0;
    for (int j = 0; j < num_points; j++)
    {
      // find each centers and add their points value together

      sum[k] += datastore[j][k];
    }
    global_center[k] = sum[k] / num_points;
  }

  // generate new centers base on the Mean of each group of datapoints
  for (int i = 0; i < num_attributes; i++)
  {
    global_center[i] = sum[i] / num_points;
    // cout << "global_center  : " << global_center[i] << endl;
  }


  for (i = 0; i < num_attributes; i++)
  {
    for (j = 0; j < num_points; j++)
    {
      S_total += ((datastore[j][i] - global_center[i]) * (datastore[j][i] - global_center[i]));

    }

  }
  S_between = S_total - S_w;

  Calinski_Harabasz_score = ((S_between / S_w) * ((num_points - num_centers) / (num_centers - 1.0)));
  //cout << "\nCalinski_Harabasz_score: " << Calinski_Harabasz_score << endl;

  delete[] sum;
  delete[]global_center;
  return Calinski_Harabasz_score;
}

void z_score(double** temp, double** datastore, double** z_value, double* mean, double num_points, int num_attributes)
{
  int new_max = 1;
  int new_min = 0;
  double* stdev = new double[num_attributes];


  for (int i = 0; i < num_attributes; i++)
  {
    mean[i] = 0;
    for (int j = 0; j < num_points; j++)
    {
      mean[i] += temp[j][i];

    }
    mean[i] = mean[i] / num_points;
    // cout << " col " << i + 1 << "mean:" << mean[i] << endl;
  }


  for (int i = 0; i < num_attributes; i++)
  {
    stdev[i] = 0.0;

    for (int j = 0; j < num_points; j++)
    {
      stdev[i] += (temp[j][i] - mean[i]) * (temp[j][i] - mean[i]);
    }
    stdev[i] = stdev[i] / num_points;
    stdev[i] = sqrt(stdev[i]);
    //avoid division by 0  
    if (fabs(stdev[i]) < DBL_EPSILON)
    {
      stdev[i] = 1;
    }
    // cout << "stdev: col " << i + 1 << " " << stdev[i] << endl;
  }
  for (int j = 0; j < num_points; j++)
  {
    for (int i = 0; i < num_attributes; i++)
    {

      z_value[j][i] = (temp[j][i] - mean[i]) / stdev[i];
      datastore[j][i] = z_value[j][i];

    }

  }

  cout << "Normalization : Z_ Score" << endl;
  delete[] stdev;

}
void min_max_normalization(double** temp, double** datastore, double** value, double* max, double* min, double num_points, int num_attributes)
{
  // cout << "Normalization : min_max_normalization" << endl;

  double new_max = 1.0;
  double new_min = 0.0;

  /* test dataset
  for (i = 0; i < num_points; i++)
  {
    for (int j = 0; j < num_attributes; j++)
    {
      cout << " " << temp[i][j];

    }
    cout << endl;
  }
*/
  for (int i = 0; i < num_attributes; i++)
  {
    max[i] = 0;
    min[i] = DBL_MAX;
    for (int j = 0; j < num_points; j++)
    {
      if (temp[j][i] < min[i])
      {
        min[i] = temp[j][i];

      }
      if (temp[j][i] > max[i])
      {
        max[i] = temp[j][i];
      }

    }
    // avoid division by 0
    if (fabs(max[i] - min[i]) < DBL_EPSILON)
    {
      max[i] = 1;
      min[i] = 0;
    }
    // cout << " col " << i + 1 << "min ::" << min[i] << " Max: " << max[i] << endl;
  }


  for (int i = 0; i < num_attributes; i++)
  {

    for (int j = 0; j < num_points; j++)
    {
      value[j][i] = (temp[j][i] - min[i]) / (max[i] - min[i]);
      datastore[j][i] = value[j][i];
    }
  }


}
void random_select(double** temp, double** clusters, int num_centers, int num_points, int num_attributes)
{
 // cout << "Initialization: random_select " << endl;
  // Uniformly random
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> select(0, num_points - 1);

  // generate first set of  random cluster centers 
  for (int i = 0; i < num_centers; i++)
  {
    int rand_points = select(gen);
    for (int j = 0; j < num_attributes; j++)
    {
      clusters[i][j] = (double)temp[rand_points][j];
    }
  }

}
void random_partition(int* labels, double** datastore, double** partition_centers, int num_centers, int num_points, int num_attributes)
{
  cout << "Initialization: random_partition " << endl;

  int i;
  // Uniformly random
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> select(0, num_centers - 1);

  for (i = 0; i < num_points; i++)
  {
    int rand_points = select(gen);
    labels[i] = rand_points;

  }
  getcenters(labels, datastore, partition_centers, num_centers, num_points, num_attributes);
}
void maximin(double** temp, double** clusters, int num_centers, int num_points, int num_attributes)
{
  //  cout << "Initialization: maximin " << endl;

  double dis_max, dis_store;
  int  label, index;
  double max_distance = 0.0;
  // delcare 2d array to store distance between centers and points
  int* index_dis = new int[num_points];
  double* distance = new double[num_points];


  // Uniformly random
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> select(0, num_points - 1);

  // generate first random cluster centers 
  for (int i = 0; i < 1; i++)
  {
    int rand_points = select(gen);
    for (int j = 0; j < num_attributes; j++)
    {
      clusters[i][j] = (double)temp[rand_points][j];
    }
  }
  // compute distance to first center
  for (int i = 0; i < num_points; i++)
  {
    distance[i] = getdistance(temp[i], clusters[0], num_attributes);
    index_dis[i] = 0;
  }
  //  compute distance of other centers
  for (int j = 1; j < num_centers; j++)
  {
    dis_max = 0;
    label = 0;

    for (int i = 0; i < num_points; i++)
    {
      //find max distance
      if (distance[i] <= dis_max)
      {
        continue;
      }
      //compute  distance from  passed centers
      while (index_dis[i] < j - 1)
      {
        index_dis[i] += 1;
        index = index_dis[i];
        dis_store = getdistance(temp[i], clusters[index], num_attributes);

        //update the min distance value if find smaller one 
        if (dis_store < distance[i])
        {
          distance[i] = dis_store;
          //break loop if it smaller than current max distance
          if (dis_store < dis_max)
          {
            break;
          }
        }
      }
      //find new max distance value and label it
      if (distance[i] > dis_max)
      {
        dis_max = distance[i];
        label = i;
      }

    }
    distance[label] = 0.0;
    //assign the initial cluster centers
    for (int k = 0; k < num_attributes; k++)
    {
      clusters[j][k] = temp[label][k];
    }

  }
  delete[] distance;
  delete[] index_dis;

}
// funtion to calculate the distance between datapoints and centers
double getdistance(double datapoints[], double centers[], int num_attributes)
{

  double sum = 0.0;
  for (int i = 0; i < num_attributes; i++)
  {
    // the total distance of datapoints with its attributes
    sum += ((datapoints[i] - centers[i]) * (datapoints[i] - centers[i]));
  }
  return sum;
}

//cluster assignments to group datapoints and lable them close to which centers(K)
void cluster(double** datastore, double** cluster_centers, int* labels, int num_centers, int num_points, int num_attributes)
{
  int i, j;
  double min_distance = 0.0;
  // delcare 2d array to store distance between centers and points
  double** distance;
  distance = new double* [num_points];
  for (i = 0; i < num_points; i++)
  {
    distance[i] = new double[num_centers];
  }

  // get distance of each row of point, compare to centers, find the closest one and lable the center num.  
  for (i = 0; i < num_points; i++)
  {
    min_distance = DBL_MAX;
    for (j = 0; j < num_centers; j++)
    {
      distance[i][j] = getdistance(datastore[i], cluster_centers[j], num_attributes);

      if (distance[i][j] < min_distance)
      {
        min_distance = distance[i][j];
        labels[i] = j;
      }

    }

  }
  for (int i = 0; i < num_points; i++)
  {
    delete[] distance[i];
  }
  delete[] distance;

}

//get SSE result 
double getSSE(int* labels, double** datapoints, double** centers, int num_centers, int num_points, int num_attributes)
{
  int i, j;
  double sum = 0.0;
  // base on labels[], calculate the total distance of each CENTERS
  for (i = 0; i < num_centers; i++)
  {
    for (j = 0; j < num_points; j++)
    {

      if (i == labels[j])
      {
        // cout<<"labels; "<<i<<endl;
        sum += getdistance(datapoints[j], centers[i], num_attributes);

      }
    }

  }

  return sum;

}
void singleton_cluster(int* labels, double** datapoints, double** centers, int num_centers, int num_points, int num_attributes)
{
  int i, j;
  //delcare a dynamic 1d array to compare the distance within the center
  double* distance = new double[num_points];
  for (i = 0; i < num_centers; i++)
  {
    for (j = 0; j < num_points; j++)
    {
      //locate the center
      if (i == labels[j])
      {
        //get distance between each datapoints and center
        distance[j] = getdistance(datapoints[j], centers[i], num_attributes);

        // if the distance are zero, there is no other datapoints, make it to be the new centers      
        if (distance[j] == 0)
        {
          for (int ii = 0; ii < num_attributes; ii++)
          {
            centers[i][ii] = datapoints[j][ii];
          }
        }
      }
    }
  }
}
// generate the cluster centers
void getcenters(int* labels, double** datastore, double** new_centers, int num_centers, int num_points, int num_attributes)
{
  int i, j, k, count;
  //declare dynamica 2d-array for storing  centers
  double** sum;
  sum = new double* [num_centers];
  for (i = 0; i < num_centers; i++)
  {
    sum[i] = new double[num_attributes];

  }
  //clean the array to prevent data conflict 
  for (i = 0; i < num_centers; i++)
  {
    for (j = 0; j < num_attributes; j++)
    {
      sum[i][j] = 0;
    }
  }

  // calculate the total of points under each centers
  for (i = 0; i < num_centers; i++)
  {
    // to record the num of datapoints under centers
    count = 0;
    for (j = 0; j < num_points; j++)
    {
      // find each centers and add their points value together
      if (i == labels[j])
      {
        for (k = 0; k < num_attributes; k++)
        {
          sum[i][k] += datastore[j][k];

        }
        count++;
      }
    }
    // generate new centers base on the Mean of each group of datapoints
    for (k = 0; k < num_attributes; k++)
    {
      new_centers[i][k] = sum[i][k] / count;
    }
  }

  for (i = 0; i < num_centers; i++)
  {
    delete[] sum[i];
  }
  delete[] sum;
}
