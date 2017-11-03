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

bool adagrad = false;
bool top100parallel = false;
bool DEBUG = false;

typedef vector<vector<float>> FloatMatrix;
typedef vector<vector<int>> IntMatrix;

string base_name;

int updates = 0;
int noupdates = 0;

int NUSER = 0, NITEMS = 0;

vector<vector<int>> tracks_in_playlist;
vector<vector<int>> test;
vector<int> ttracks;
vector<int> tplaylists;

class Sample {
public:
    int u, i, j;
    Sample(int u, int i, int j) : u(u), i(i), j(j) {};
};

class TopNElements {
public:
    int size;
    vector<float> elems;
    vector<int> idxs;
    TopNElements(int size) : size(size), elems(size), idxs(size) {
        idxs.assign(size, 0);
        elems.assign(size, -1);
    }
    void push(float e, int index) {
        if (e <= elems[size-1])
            return;
        int i = size -1;
        while (i > 0 && e > elems[i-1]) {
            elems[i] = elems[i-1];
            idxs[i] = idxs[i-1];
            i--;
        }
        elems[i] = e;
        idxs[i] = index;
    }

    void sorted(vector<float> &elements, vector<int> &indexes) {
        vector<pair<int, float>> V;
        for (int i = 0; i < size; i++)
            V.push_back({idxs[i], elems[i]});
        sort(V.begin(), V.end());

        elements.resize(size);
        indexes.resize(size);
        for (int i = 0; i < size; i++){
            elements[i] = V[i].second;
            indexes[i] = V[i].first;
        }
    }
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

void get_target_tracks(vector<int> &ttracks) {
    ifstream fs (base_name + "/target_tracks.txt");

    if (!fs) {
        printf("Error opening target_tracks.txt\n");
        exit(0);
    }

    ttracks.clear();

    int N;
    fs >> N;

    for (int i = 0; i < N; i++) {
        int tmp;
        fs >> tmp;
        ttracks.push_back(tmp);
    }
    fs.close();
}

void get_target_playlists(vector<int> &tplaylists) {
    ifstream fs (base_name + "/target_playlists.txt");

    if (!fs) {
        printf("Error opening target_playlists.txt\n");
        exit(0);
    }

    tplaylists.clear();

    int N;
    fs >> N;

    for (int i = 0; i < N; i++) {
        int tmp;
        fs >> tmp;
        tplaylists.push_back(tmp);
    }
    fs.close();
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

void read_test(vector<vector<int>>& test) {
    ifstream input (base_name + "/test.txt");

    if (!input) {
        printf("Error opening test.txt\n");
        exit(0);
    }
    string sizes, rows, cols;

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);

    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);

    int testusers, testitems;
    sizestream >> testusers >> testitems;
    test.resize(max(testusers, NUSER));

    int r,c;
    while (rstream >> r) {
        cstream >> c;
        test[r].push_back(c);
    }

    for (int i = 0; i < NUSER; i++)
        sort(test[i].begin(), test[i].end());
}

void predict_5(IntMatrix& urm, FloatMatrix& S, IntMatrix& indexes, vector<int>& ttracks, int pl_num, vector<int>& pred) {
    TopNElements topn(5);
    for (int tr_id : ttracks) {
        int i_ptr = 0;
        float ret = 0.0;
        if (find(urm[pl_num].begin(), urm[pl_num].end(), tr_id) != urm[pl_num].end())
            continue;
        for (int u_idx : urm[pl_num]) {
            while (i_ptr < indexes[tr_id].size() && indexes[tr_id][i_ptr] < u_idx)
                i_ptr++;
            if (i_ptr < indexes[tr_id].size() && indexes[tr_id][i_ptr] == u_idx)
                ret += S[tr_id][i_ptr];

            if (i_ptr == indexes[tr_id].size() && indexes[tr_id][i_ptr] <= u_idx)
                break;
        }

        topn.push(ret, tr_id);
    }

    for (int i = 0; i < 5; i++) {
        pred[i] = topn.idxs[i];
    }

}

float map_precision(IntMatrix& test, int pl_num, vector<int>& predictions) {
    float correct = 0.0, ap = 0.0;
    for (int i = 0; i < predictions.size(); i++) {
        if (find(test[pl_num].begin(), test[pl_num].end(), predictions[i]) != test[pl_num].end()) {
            correct += 1;
            ap += correct/(i+1);
        }
    }
    return ap / predictions.size();
}

float computeMAP(IntMatrix& urm, IntMatrix& test, FloatMatrix& S, IntMatrix& indexes, vector<int>& ttracks, vector<int>& tplaylists) {
    cout << "Computing MAP... " << flush;
    int count = 0;
    double cumsum = 0.0;

    vector<int> predictions;
    predictions.resize(5);


    sort(ttracks.begin(), ttracks.end());
    sort(tplaylists.begin(), tplaylists.end());

    #pragma omp parallel for reduction(+:cumsum)
    for (int i = 0; i < tplaylists.size(); i++) {
        int pl_num = tplaylists[i];
        predict_5(urm, S, indexes, ttracks, pl_num, predictions);
        cumsum += map_precision(test, pl_num, predictions);
    }

    return cumsum / tplaylists.size();
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
    vector<float> adagrad_weights;
    adagrad_weights.assign(NITEMS, 0.0);

    for (int it = 0; it < iterations; it++) {
        cout << "Iteration: " << it << endl;
        vector<Sample> samples;
        vector<pair<float, int>> results_j;

        collect_samples(1, samples, tracks_in_playlist);

        updates += samples.size();

        random_shuffle(samples.begin(), samples.end());

        for (int count = 0; count < samples.size(); count++) {
            int u,i,j;
            u = samples[count].u;
            i = samples[count].i;
            j = samples[count].j;
            // sample(u, i, j, urm);

            float x_pred;
            if (top100parallel) {
                results_j.clear();
                results_j.resize(100);

                #pragma omp parallel for
                for (int ii = 0; ii < 100; ii++) {
                    int jj = sample_negative(u, urm);
                    results_j[ii] = make_pair(predict_xuij(u, i, jj, urm, S, indexes), jj);
                }

                sort(results_j.begin(), results_j.end());
                x_pred = results_j[0].first;
                j = results_j[0].second;
            }
            else {
                x_pred = predict_xuij(u, i, j, urm, S, indexes);
                //cout << "x_pred = " << x_pred << endl;
                int cutoff = 100;
                while (x_pred > 0.0 && cutoff--) {
                    j = sample_negative(u, urm);
                    x_pred = predict_xuij(u, i, j, urm, S, indexes);
                    //cout << "Resampled. x_pred = " << x_pred << endl;
                }
            }




            if (x_pred <= 0.1) {
                float z = sigmoid(x_pred);

                adagrad_weights[i] += z*z;
                adagrad_weights[j] += z*z;

                int psi = 0, psj = 0;
                for (int l = 0; l < urm[u].size(); l++) {
                    int idx = urm[u][l];
                    while (psi < indexes[i].size() && idx > indexes[i][psi])
                        psi++;
                    while (psj < indexes[j].size() && idx > indexes[j][psj])
                        psj++;

                    if (psi < indexes[i].size() && idx != i && idx == indexes[i][psi]) {
                        if (adagrad)
                            S[i][psi] += (alpha / sqrt(adagrad_weights[i] + 0.000001))*(z - reg_positive*S[i][psi]);
                        else
                            S[i][psi] += (alpha)*(z - reg_positive*S[i][psi]);
                    }

                    if (psj < indexes[j].size() && idx != j && idx == indexes[j][psj]){
                        if (adagrad)
                            S[j][psj] += (alpha / sqrt(adagrad_weights[j] + 0.000001))*(-z - reg_negative*S[j][psj]);
                        else
                            S[j][psj] += (alpha)*(-z - reg_negative*S[j][psj]);
                    }
                }
            }
            else {
                noupdates ++;
            }

        }
        cout << "MAP: " << fixed << setprecision(5) << computeMAP(urm, test, S, indexes, ttracks, tplaylists) << endl;

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

    cout << "Reading target tracks" << endl;
    get_target_tracks(ttracks);

    cout << "Reading target playlists" << endl;
    get_target_playlists(tplaylists);

    cout << "Reading test" << endl;
    read_test(test);

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
