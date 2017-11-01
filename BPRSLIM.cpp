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
#include <assert.h>

using namespace std;

const int SAMPLES_PER_EPOCH = 10000;

typedef vector<vector<float>> FloatMatrix;
typedef vector<vector<int>> IntMatrix;

string base_name;

int updates = 0;

int NUSER = 0, NITEMS = 0;

vector<vector<int>> tracks_in_playlist;


class Sample {
public:
    int u, i, j;
    Sample(int u, int i, int j) : u(u), i(i), j(j) {};
};







float gen_rand() {
    return ((float) rand() / (RAND_MAX));
}



void print_similarity(ofstream& output, vector<vector<float>>& similarity, vector<vector<int>>& indexes) {
    // Print rows
    output << NITEMS << " " << NITEMS << endl;
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

void read_similarity_matrix(vector<vector<float>> &similarity, vector<vector<int>> &indexes, int read_values) {
    string weights, rows, cols, data, sizes;
    ifstream input (base_name + "/similarity.txt");

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);


    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);
    stringstream dstream;
    if (read_values) {
        getline(input, data);
        dstream << data;
    }


    int size1, size2;
    sizestream >> size1 >> size2;

    if (NITEMS != size1 || NITEMS != size2) {
        cout << "Dimensions mismatch: URM has size: " << NUSER << "x" << NITEMS << " but similarity has size: " << size1 << "x" << size2 << endl;
        exit(0);
    }

    indexes.resize(NITEMS);
    similarity.resize(NITEMS);

    int r, c;
    float d;
    while (rstream >> r) {
        cstream >> c;

        indexes[r].push_back(c);
        if (read_values) {
            dstream >> d;
            similarity[r].push_back(d);
        }
        else
            similarity[r].push_back(0);
    }

    for (int i = 0; i < NITEMS; i++)
        sort(indexes[i].begin(), indexes[i].end());
}




inline int sample_negative(int user, vector<vector<int>>& urm) {
    int j = rand() % (NITEMS);
    while (find(urm[user].begin(), urm[user].end(), j) != urm[user].end())
        j = rand() % (NITEMS);

    return j;
}

void sample(int& u, int& i, int& j, vector<vector<int>>& urm) {
    u = rand()%(urm.size());
    while (urm[u].size() == 0)
        u = rand()%(urm.size());

    i = urm[u][rand() % (urm[u].size())];
    j = sample_negative(u, urm);
}



void collect_samples(int numiterations, vector<Sample>& samples, IntMatrix& urm) {
    samples.reserve(numiterations * SAMPLES_PER_EPOCH);
    for (int u = 0; u < urm.size(); u++) {
        for (int i : urm[u]) {
            samples.push_back(Sample(u,i,sample_negative(u, urm)));
        }

    }
}

void build_similarity(vector<Sample>& samples, IntMatrix& urm, IntMatrix& indexes, FloatMatrix& similarity) {
    vector<set<int>> S;
    S.resize(NITEMS);
    int count = 0;
    sort(samples.begin(), samples.end(), [](Sample& s1, Sample& s2) {return s1.u < s2.u;});
    for (auto& s : samples) {
        for (int l : urm[s.u]) {
            if (l != s.i) {
                count ++;
                S[s.i].insert(l);
            }

            if (l != s.j) {
                count ++;
                S[s.j].insert(l);
            }
        }
    }

    similarity.resize(NITEMS);
    indexes.resize(NITEMS);

    for (int i = 0; i < NITEMS; i++) {
        indexes[i].insert(indexes[i].end(), S[i].begin(), S[i].end());
        similarity[i].assign(S[i].size(), 0);
    }
}

float predict_xuij(int u, int i, int j, vector<vector<int>>& urm, vector<vector<float>> &S, vector<vector<int>> &indexes) {
    int psi = 0, psj = 0;
    float count = 0.0;

    for (int k = 0; k < urm[u].size(); k++) {
        // Advance psi & psj
        while (psi < indexes[i].size() && urm[u][k] > indexes[i][psi])
            psi++;
        while (psj < indexes[j].size() && urm[u][k] > indexes[j][psj])
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
        vector<Sample> samples;
        collect_samples(1, samples, tracks_in_playlist);

        updates += samples.size();

        random_shuffle(samples.begin(), samples.end());

        for (int count = 0; count < samples.size(); count++) {
            int u,i,j;
            u = samples[count].u;
            i = samples[count].i;
            j = samples[count].j;
            // sample(u, i, j, urm);

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

                if (psi < indexes[i].size() && idx != i && idx == indexes[i][psi]) {
                    S[i][psi] += alpha*(z - reg_positive*S[i][psi]);
                    // updates++;
                }

                if (psj < indexes[j].size() && idx != j && idx == indexes[j][psj]){
                    S[j][psj] += alpha*(-z - reg_negative*S[j][psj]);
                    // updates++;
                }
            }
        }
    }
}



int main(int argc, char *argv[]) {
    srand(time(NULL));
    vector<vector<float>> similarity;
    vector<vector<int>> indexes;
    int read_values = 0;

    int numiterations = 70;
    float reg_positive = 0.1, reg_negative = 0.01, alpha = 0.01;

    if (argc >= 2)
        base_name = string(argv[1]);
    else {
        cout << "Usage: " << argv[0] << " <folder> [<numiterations> <alpha> <reg_positive> <reg_negative>] [<read_values>=0/1]" << endl;
        exit(0);
    }
    if (argc >= 6) {
        numiterations = atoi(argv[2]);
        alpha = atof(argv[3]);
        reg_positive = atof(argv[4]);
        reg_negative = atof(argv[5]);
    }

    if (argc >= 7) {
        read_values = atoi(argv[6]);
    }

    cout << "Reading tracks in playlist..." << endl;
    read_tracks_in_playlist(tracks_in_playlist);

    cout << "Reading similarity matrix" << endl;
    read_similarity_matrix(similarity, indexes, read_values);
    vector<Sample> samples;
    collect_samples(1, samples, tracks_in_playlist);
    //cout << "Building similarity.." << endl;
    //build_similarity(samples, tracks_in_playlist, indexes, similarity);

    cout << "Starting" << endl;


    clock_t init, finish;

    init = clock();
    BPRSLIM(tracks_in_playlist, similarity, indexes, numiterations, alpha, reg_positive, reg_negative);

    finish = clock();
    double elapsedTime = double(finish -init)/ CLOCKS_PER_SEC;
    cout << "iterations: " << std::fixed << std::setprecision(3) << 1.0 * updates / elapsedTime << endl;


    ofstream output;
    output.open (base_name + "/similarity_bpr.txt");

    cout << "Printing similarity matrix..." << endl;
    print_similarity(output, similarity, indexes);
}
