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
#include <omp.h>

using namespace std;

typedef vector<vector<float>> FloatMatrix;
typedef vector<vector<int>> IntMatrix;

string base_name;
string sgd_method;
string similarities_bitmask;

int updates = 0;

int NUSER = 0, NITEMS = 0;


class Sample {
public:
    int u, i, j;
    Sample(int u, int i, int j) : u(u), i(i), j(j) {};
};







float gen_rand() {
    return ((float) rand() / (RAND_MAX));
}



void print_parameters(ofstream& output, FloatMatrix& parameters, string similarities_bitmask) {
    // a parameter for each line
    for (int p = 0; p < similarities_bitmask.size(); p++) {
        if (similarities_bitmask[p] == '1') {
            for (int u = 0; u < parameters.size(); u++) {
                output << parameters[u][p] << " ";
            }
            output << endl;
        }
    }
}


void read_tracks_in_playlist(FloatMatrix& tracks_in_playlist, IntMatrix& tracks_in_playlist_indexes,
                            FloatMatrix& parameters, string similarities_bitmask, vector<int>& target_playlists) {
    ifstream input (base_name + "/tracks_in_playlist.txt");

    if (!input) {
        printf("Error opening tracks_in_playlist.txt\n");
        exit(0);
    }
    string sizes, rows, cols, data, tpl;

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);
    getline(input, data);
    getline(input, tpl);

    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);
    stringstream datastream(data);
    stringstream tpstream(tpl);

    sizestream >> NUSER >> NITEMS;
    tracks_in_playlist.resize(NUSER);
    tracks_in_playlist_indexes.resize(NUSER);

    int r,c;
    float v;
    while (rstream >> r) {
        cstream >> c;
        datastream >> v;

        tracks_in_playlist[r].push_back(v);
        tracks_in_playlist_indexes[r].push_back(c);
    }

    for (int i = 0; i < NUSER; i++)
        sort(tracks_in_playlist_indexes[i].begin(), tracks_in_playlist_indexes[i].end());

    // Initialize every playlist with parameters of value 1
    int num_similarities = similarities_bitmask.size();
    for (int i = 0; i < NUSER; i++) {
        vector<float> current_user_parameters;
        for (int s = 0; s < similarities_bitmask.size(); s++) {
            if (similarities_bitmask[s] == '1') {
                current_user_parameters.push_back(1);
            } else {
                current_user_parameters.push_back(0);
            }
        }
        parameters.push_back(current_user_parameters);
    }

    int pl_id;
    while (tpstream >> pl_id) {
        target_playlists.push_back(pl_id);
    }
}

void read_similarity_matrix(vector<FloatMatrix>& similarity_vector, vector<IntMatrix>& indexes_vector,
        FloatMatrix& similarity_row_sums, int num_sim) {
    string weights, rows, cols, data, sizes;
    ifstream input (base_name + "/similarity_" + to_string(num_sim) + ".txt");

    getline(input, sizes);
    getline(input, rows);
    getline(input, cols);
    getline(input, data);


    stringstream sizestream(sizes);
    stringstream rstream(rows);
    stringstream cstream(cols);
    stringstream dstream(data);


    int size1, size2;
    sizestream >> size1 >> size2;

    if (NITEMS != size1 || NITEMS != size2) {
        cout << "Dimensions mismatch: URM has size: " << NUSER << "x" << NITEMS << " but similarity has size: " << size1 << "x" << size2 << endl;
        exit(0);
    }

    IntMatrix indexes;
    FloatMatrix similarity;
    vector<float> row_sums;

    indexes.resize(NITEMS);
    similarity.resize(NITEMS);
    row_sums.assign(NITEMS, 0);

    int r, c;
    float d;
    while (rstream >> r) {
        cstream >> c;
        dstream >> d;

        indexes[r].push_back(c);
        similarity[r].push_back(d);
        row_sums[r] += d;
    }

    for (int i = 0; i < NITEMS; i++)
        sort(indexes[i].begin(), indexes[i].end());

    similarity_vector[num_sim] = similarity;
    indexes_vector[num_sim] = indexes;
    similarity_row_sums[num_sim] = row_sums;
}




inline int sample_negative(int user, IntMatrix& urm) {
    int j = rand() % (NITEMS);
    while (find(urm[user].begin(), urm[user].end(), j) != urm[user].end())
        j = rand() % (NITEMS);

    return j;
}

void sample(int& u, int& i, int& j, IntMatrix& urm) {
    u = rand()%(urm.size());
    while (urm[u].size() == 0)
        u = rand()%(urm.size());

    i = urm[u][rand() % (urm[u].size())];
    j = sample_negative(u, urm);
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

float predict_xuij(int u, int i, int j, FloatMatrix& urm, IntMatrix& urm_indexes,
                FloatMatrix& S, IntMatrix& indexes, vector<float>& row_sums) {
    int psi = 0, psj = 0;
    float count = 0.0;

    for (int k = 0; k < urm_indexes[u].size(); k++) {
        // Advance psi & psj
        while (psi < indexes[i].size() && urm_indexes[u][k] > indexes[i][psi])
            psi++;
        while (psj < indexes[j].size() && urm_indexes[u][k] > indexes[j][psj])
            psj++;

        if (psi < indexes[i].size() && urm_indexes[u][k] == indexes[i][psi])
            count += urm[u][k] * S[i][psi];// / row_sums[psi];
        if (psj < indexes[j].size() && urm_indexes[u][k] == indexes[j][psj])
            count -= urm[u][k] * S[j][psj];// / row_sums[psj];
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

void get_schwifty(FloatMatrix& urm, IntMatrix& urm_indexes,
                vector<FloatMatrix>& similarity_vector, vector<IntMatrix>& indexes_vector, FloatMatrix& similarity_row_sums,
                FloatMatrix& parameters, string similarities_bitmask,  vector<int>& target_playlists,
                int iterations=1, float alpha = 0.1, float theta = 0.01, float x_thresh = 100, float gamma = 0.9) {
    
    int num_similarities = similarities_bitmask.size();

    // For each playlist, do several iterations...
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int ui = 0; ui < target_playlists.size(); ui++) {
        int u = target_playlists[ui];

        vector<Sample> samples;

        samples.reserve(urm_indexes[u].size() * iterations);
        for (int i : urm_indexes[u]) {
            for (int it = 0; it < iterations; it++) {
                samples.push_back(Sample(u,i,sample_negative(u, urm_indexes)));
            }
        }

        updates += samples.size();
        random_shuffle(samples.begin(), samples.end()); 

        // Used only with adagrad and adadelta
        vector<float> past_squared_gradients_cumsum;
        past_squared_gradients_cumsum.assign(num_similarities, 0);

        // Used only with  adadelta
        vector<float> past_squared_delta_param_cumsum;
        past_squared_delta_param_cumsum.assign(num_similarities, 0);
        
        int num_up = 0;
        for (int count = 0; count < samples.size(); count++) {
            int u,i,j;
            u = samples[count].u;
            i = samples[count].i;
            j = samples[count].j;
            // sample(u, i, j, urm);

            vector<float> xs;
            xs.assign(num_similarities, 0);
            float weighted_x = 0;
            for (int s = 0; s < similarities_bitmask.size(); s++) {
                if (similarities_bitmask[s] == '1') {
                    float x = predict_xuij(u, i, j, urm, urm_indexes, similarity_vector[s], indexes_vector[s], similarity_row_sums[s]);
                    xs[s] = x;
                    weighted_x += parameters[u][s] * x;
                }
            }

            if (weighted_x > x_thresh) {
                continue;
            }

            float K = (1 - sigmoid(weighted_x));

            for (int p = 0; p < similarities_bitmask.size(); p++) {
                if (similarities_bitmask[p] == '1') {
                    float gradient = K * xs[p] - 2 * theta * parameters[u][p];

                    if (sgd_method.compare("adagrad") == 0) {
                        float epsilon = 0.00001;
                        parameters[u][p] += (alpha / sqrt(past_squared_gradients_cumsum[p] + epsilon)) * gradient;
                        past_squared_gradients_cumsum[p] += pow(gradient, 2);
                    } else if (sgd_method.compare("adadelta") == 0) {
                        float epsilon = 0.00001;
                        float delta_param = (sqrt(past_squared_delta_param_cumsum[p] + epsilon) / sqrt(past_squared_gradients_cumsum[p] + epsilon)) * gradient;
                        parameters[u][p] += delta_param;
                        past_squared_gradients_cumsum[p] = gamma * past_squared_gradients_cumsum[p] + (1 - gamma) * pow(gradient, 2);
                        past_squared_delta_param_cumsum[p] = gamma * past_squared_delta_param_cumsum[p] + (1 - gamma) * pow(delta_param, 2);
                    } else {
                        parameters[u][p] += alpha * gradient;
                    }
                }
            }

            num_up++;
        }

        if (ui % 100 == 0) {
            cout << ui << " out of " << target_playlists.size() << endl;
        }
    }
}



int main(int argc, char *argv[]) {
    srand(time(NULL));
    vector<FloatMatrix> similarity_vector;
    vector<IntMatrix> indexes_vector;
    FloatMatrix similarity_row_sums;

    FloatMatrix parameters;
    IntMatrix tracks_in_playlist_indexes;
    FloatMatrix tracks_in_playlist;
    vector<int> target_playlists;

    int numiterations = 70;
    sgd_method = "sgd";
    float alpha = 0.1;
    float gamma = 0.9;
    float theta = 0.01;
    float x_thresh = 100;

    if (argc >= 3) {
        base_name = string(argv[1]);
        similarities_bitmask = string(argv[2]);
    }
    else {
        cout << "Usage:\n" << argv[0] << " <folder> <similarities_bitmask> [<sgd_method> <numiterations>]\n"
            "\tif sgd_method == 'sgd': [<alpha> <theta> <x_thresh>]\n"
            "\tif sgd_method == 'adagrad': [<alpha> <theta> <x_thresh>]"
            "\tif sgd_method == 'adadelta': [<gamma> <theta> <x_thresh>]" << endl;
        exit(0);
    }

    if (argc >= 7) {
        sgd_method = string(argv[3]);
        numiterations = atoi(argv[4]);
        if (sgd_method.compare("adadelta") == 0) gamma = atof(argv[5]);
        else alpha = atof(argv[5]);
        theta = atof(argv[6]);
        x_thresh = atof(argv[7]);
    }

    cout << "Reading tracks in playlist..." << endl;
    read_tracks_in_playlist(tracks_in_playlist, tracks_in_playlist_indexes, parameters, similarities_bitmask, target_playlists);

    cout << "Reading similarity matrices..." << endl;

    int num_similarities = similarities_bitmask.size();
    similarity_vector.resize(num_similarities);
    indexes_vector.resize(num_similarities);
    similarity_row_sums.resize(num_similarities);

    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < similarities_bitmask.size(); i++) {
        char ch = similarities_bitmask[i];
        if (ch == '1') {
            read_similarity_matrix(similarity_vector, indexes_vector, similarity_row_sums, i);
            cout << "Read similarity number " << i << endl;
        }
    }

    cout << "Starting get schwifty!" << endl;

    clock_t init, finish;

    init = clock();
    get_schwifty(tracks_in_playlist, tracks_in_playlist_indexes, similarity_vector, indexes_vector, similarity_row_sums, 
        parameters, similarities_bitmask, target_playlists,
        numiterations, alpha, theta, x_thresh, gamma);

    finish = clock();
    double elapsedTime = double(finish -init)/ CLOCKS_PER_SEC;
    cout << "iterations: " << std::fixed << std::setprecision(3) << 1.0 * updates / elapsedTime << endl;


    ofstream output;
    output.open (base_name + "/playlist_params.txt");

    cout << "Printing playlist_params..." << endl;
    print_parameters(output, parameters, similarities_bitmask);
}
