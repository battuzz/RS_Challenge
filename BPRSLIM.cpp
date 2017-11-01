#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <cmath>
#include <sstream>
#include <ctime>
#include <iomanip>

using namespace std;

string base_name;

int updates = 0;
int SIZE = 0;
int NUSER = 0, NITEMS = 0;

vector<vector<int>> tracks_in_playlist;

float gen_rand() {
    return ((float) rand() / (RAND_MAX));
}



void print_similarity(ofstream& output, vector<vector<float>>& similarity, vector<vector<int>>& indexes) {
    // Print rows
    output << SIZE << " " << SIZE << endl;
    for (int i = 0; i < similarity.size(); i++)
        for (int j = 0; j < similarity[i].size(); j++)
            output << i << " ";
    output << endl;

    // Print cols
    for (int i = 0; i < similarity.size(); i++)
        for (int j = 0; j < similarity[i].size(); j++) {
            output << indexes[i][j] << " ";
        }
    output << endl;

    // Print data
    for (int i = 0; i < similarity.size(); i++) {
        for (int j = 0; j < similarity[i].size(); j++)
            output << similarity[i][j] << " ";
    }
    output << endl;
}


void read_tracks_in_playlist(vector<vector<int>>& tracks_in_playlist) {
    ifstream input (base_name + "/tracks_in_playlist.txt");

    if (!input) {
        printf("Error opening tracks_in_playlist.txt\n");
        exit(0);
    }
    string sizes, rows, cols;

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);

    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);

    sizestream >> NUSER >> NITEMS;
    tracks_in_playlist.resize(NUSER);

    int r,c;
    while (rstream >> r) {
        cstream >> c;
        tracks_in_playlist[r].push_back(c);
    }

    for (int i = 0; i < NUSER; i++)
        sort(tracks_in_playlist[i].begin(), tracks_in_playlist[i].end());
}

void read_similarity_matrix(vector<vector<float>> &similarity, vector<vector<int>> &indexes) {
    string weights, rows, cols, data, sizes;
    ifstream input (base_name + "/similarity.txt");

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);
    //getline(input, data);

    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);
    //stringstream dstream(data);


    sizestream >> SIZE;
    indexes.resize(SIZE);
    similarity.resize(SIZE);

    int r, c;
    float d;
    while (rstream >> r) {
        cstream >> c;
        //dstream >> d;

        indexes[r].push_back(c);
        similarity[r].push_back(0);
    }
}




inline int sample_negative(int user, vector<vector<int>>& urm) {
    int j = rand() % (SIZE);
    while (find(urm[user].begin(), urm[user].end(), j) != urm[user].end())
        j = rand() % (SIZE);

    return j;
}

void sample(int& u, int& i, int& j, vector<vector<int>>& urm) {
    u = rand()%(urm.size());
    while (urm[u].size() == 0)
        u = rand()%(urm.size());

    i = urm[u][rand() % (urm[u].size())];
    j = sample_negative(u, urm);
}

float predict_xuij(int u, int i, int j, vector<vector<int>>& urm, vector<vector<float>> &S, vector<vector<int>> &indexes) {
    int psi = 0, psj = 0;
    float count = 0.0;

    for (int k = 0; k < urm[u].size(); k++) {
        // Advance psi & psj
        while (psi < indexes[i].size() && urm[u][k] > indexes[i][psi])
            psi++;
        while (psj < indexes[j].size() && urm[u][k] > indexes[j][psi])
            psj++;

        if (psi < indexes[i].size() && urm[u][k] == indexes[i][psi])
            count += S[i][psi];
        if (psj < indexes[j].size() && urm[u][k] == indexes[j][psj])
            count -= S[j][psj];
    }

    return count;
}

inline float sigmoid(float z) {
    if (z > 0) {
        return exp(-z) / (1.0 + exp(-z));
    }
    else {
        return 1.0/(1.0 + exp(z));
    }
}

void BPRSLIM(vector<vector<int>>& urm, vector<vector<float>> &S, vector<vector<int>> &indexes, int iterations=1, float alpha = 0.1, float reg_positive = 0.01, float reg_negative=0.001) {
    for (int it = 0; it < iterations; it++) {
        cout << "Iteration: " << it << endl;
        for (int count = 0; count < 1000000; count++) {
            int u,i,j;
            sample(u, i, j, urm);

            float x_pred = predict_xuij(u, i, j, urm, S, indexes);

            //float z = 1.0 / (1.0 + exp(x_pred));
            float z = sigmoid(x_pred);

            int psi = 0, psj = 0;
            for (int l = 0; l < urm[u].size(); l++) {
                int idx = urm[u][l];
                while (psi < indexes[i].size() && idx > indexes[i][psi])
                    psi++;
                while (psj < indexes[j].size() && idx > indexes[j][psj])
                    psj++;

                if (idx != i && idx == indexes[i][psi]) {
                    S[i][psi] += alpha*(z - reg_positive*S[i][psi]);
                    updates++;
                }

                if (idx != j && idx == indexes[j][psj]){
                    S[j][psj] += alpha*(-z - reg_negative*S[j][psj]);
                    updates++;
                }
            }
        }
    }
}



int main(int argc, char *argv[]) {
    srand(time(NULL));
    vector<vector<float>> similarity;
    vector<vector<int>> indexes;

    int numiterations = 70;
    float reg_positive = 0.1, reg_negative = 0.01, alpha = 0.01;

    if (argc >= 2)
        base_name = string(argv[1]);
    else {
        cout << "Usage: " << argv[0] << " <folder> [<numiterations> <alpha> <reg_positive> <reg_negative>]" << endl;
        exit(0);
    }
    if (argc >= 6) {
        numiterations = atoi(argv[2]);
        alpha = atof(argv[3]);
        reg_positive = atof(argv[4]);
        reg_negative = atof(argv[5]);
    }

    cout << "Reading tracks in playlist..." << endl;
    read_tracks_in_playlist(tracks_in_playlist);

    cout << "Reading similarity matrix" << endl;
    read_similarity_matrix(similarity, indexes);

    if (NITEMS != SIZE) {
        cout << "Dimensions mismatch: URM has size: " << NUSER << "x" << NITEMS << " but similarity has size: " << SIZE << "x" << SIZE << endl;
        exit(0);
    }

    cout << "Starting" << endl;



    clock_t init = clock();
    BPRSLIM(tracks_in_playlist, similarity, indexes, numiterations, alpha, reg_positive, reg_negative);

    clock_t finish = clock();
    double elapsedTime = double(finish -init)/ CLOCKS_PER_SEC;
    cout << "updates: " << std::fixed << std::setprecision(3) << 1.0 * updates / elapsedTime << endl;
    cout << "iterations: " << std::fixed << std::setprecision(3) << numiterations * 1000000.0 / elapsedTime << endl;


    ofstream output;
    output.open (base_name + "/similarity_bpr.txt");

    cout << "Printing similarity matrix..." << endl;
    print_similarity(output, similarity, indexes);
}
