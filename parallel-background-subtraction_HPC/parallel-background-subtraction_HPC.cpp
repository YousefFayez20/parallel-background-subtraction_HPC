#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <regex>
#include <cmath> // Required for std::round

#define MODE "sequential" // sequential mpi
//#define OMP
 //#define MEDIAN_FILTER

#define INPUT_PATH "Dataset"
#define OUTPUT_PATH "Output"
#define VERBOSE false
#define DEFAULT_DATASET "1"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// === Helper Functions === //

string clean_path(string path) {

	// Removes Extra Whitespace if detected.

	path.erase(0, path.find_first_not_of(" \t\r\n\""));
	path.erase(path.find_last_not_of(" \t\r\n\"") + 1);

	return path;
}

bool is_number(const string& str) {
	if (str.empty()) return false;
	string::const_iterator iterator = str.begin();
	while (iterator != str.end()) {
		if (isdigit(*iterator))
			iterator++;
		else return false;
	}
	return true;

}

void get_dataset_names(vector<pair<string, string>>& dataset_name_path) {
	string dataset_path = INPUT_PATH;
	for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
		if (entry.is_directory()) {
			for (const auto& inner_entry : fs::directory_iterator(entry.path())) {
				if (inner_entry.is_regular_file()) {
					string name, path;
					name = path = entry.path().string();
					name.erase(0, 8); // erase Dataset
					replace(name.begin(), name.end(), '\\', ' '); // replace backslash
					dataset_name_path.push_back({ name, path });
					break;
				}
			}
		}
	}
}

// Custom comparator to extract frame number from filename
bool naturalSort(const fs::directory_entry& a, const fs::directory_entry& b) {
	std::regex re(R"((\d+))");  // Matches sequences of digits
	std::smatch matchA, matchB;
	std::string nameA = a.path().filename().string();
	std::string nameB = b.path().filename().string();

	std::regex_search(nameA, matchA, re);
	std::regex_search(nameB, matchB, re);

	int numA = matchA.empty() ? 0 : std::stoi(matchA[1].str());
	int numB = matchB.empty() ? 0 : std::stoi(matchB[1].str());

	return numA < numB;
}

// === Median of Medians elper Functions === //

uchar get_pivot_val(vector<uchar>& vec, int low, int high) {
	if (high - low + 1 <= 9) { // if small size, get mean by sorting (will not increase time complexity)
		vector<uchar> temp(vec.begin() + low, vec.begin() + high + 1);
		sort(temp.begin(), temp.end());
		return temp[temp.size() / 2];
	}

	vector<uchar> medians;
	medians.reserve((high - low + 1 + 4) / 5);

	for (int i = low; i <= high; i += 5) {
		vector<uchar> temp;
		temp.reserve(5);
		// Group each 5 elements together
		for (int j = 0; j < 5 && i + j <= high; j++) {
			temp.push_back(vec[i + j]);
		}
		// Get median
		sort(temp.begin(), temp.end());
		medians.push_back(temp[temp.size() / 2]);
	}

	if (medians.size() == 1) // Return val
		return medians[0];
	else  // Recursive call to get median of medians
		return get_pivot_val(medians, 0, medians.size() - 1);
}

int partition_func(vector<uchar>& vec, int low, int high, uchar pivot) {
	int i = low;
	int j = high;

	while (i <= j) {
		while (i <= high && vec[i] < pivot) i++;
		while (j >= low && vec[j] > pivot) j--;

		if (i <= j) {
			swap(vec[i], vec[j]);
			i++;
			j--;
		}
	}

	return j; // index where elements <= pivot
}

int find_median_util(vector<uchar>& vec, int k, int low, int high) {
	if (low == high) { // Breaking condition
		return vec[low];
	}

	uchar pivot = get_pivot_val(vec, low, high); // Get pivot value
	int pivot_ind = partition_func(vec, low, high, pivot); // Get pivot index

	int length = pivot_ind - low + 1;

	if (k <= length) { // Recurisve call but find in lower portion
		return find_median_util(vec, k, low, pivot_ind);
	}
	else {  // Recurisve call but find in higher portion
		return find_median_util(vec, k - length, pivot_ind + 1, high);
	}
}

float find_median(vector<uchar> vec) {  // O(n) Algorithm using median of medians
	int n = vec.size();

	if (n % 2 == 1) { // handle odd sized vec
		return find_median_util(vec, (n + 1) / 2, 0, n - 1);
	}
	else { // handle even size vec by taking average
		uchar lower = find_median_util(vec, n / 2, 0, n - 1);
		vector<uchar> vec_copy = vec; // use vec copy
		uchar upper = find_median_util(vec_copy, n / 2 + 1, 0, n - 1);
		return (lower + upper) / 2.0;
	}
}



// === Algorithm Steps === //

bool set_directories(string& input_directory, string& output_directory) {

	string input;
	// == Dataset Names == //
	vector<pair<string, string>> dataset_name_path;
	get_dataset_names(dataset_name_path);

	if (VERBOSE) {
		cout << "Please Choose A Dataset For Your Inputes Frames Directory: \n";
		for (int i = 0; i < dataset_name_path.size(); i++) {
			cout << i + 1 << ") " << dataset_name_path[i].first << "\n";
		}

		// == Input Directory == //
		cout << "Or Enter The Path Of Your Inputes Frames Directory: \n";

		getline(cin, input);
		input = clean_path(input);
	}
	else {
		// For evaluation assume input DEFAULT_DATASET (1)
		input = DEFAULT_DATASET;
	}

	if (is_number(input)) {
		// Validating dataset number.
		int number = stoi(input) - 1;
		if (number < 0 || number >= dataset_name_path.size()) {
			cerr << "Error: Input number does not exist or is not a valid number.\n";
			return 0;
		}
		input_directory = dataset_name_path[stoi(input) - 1].second;
		output_directory = OUTPUT_PATH + dataset_name_path[stoi(input) - 1].first;
	}
	else
		input_directory = input;

	// Validating selected input directory.
	if (!fs::exists(input_directory) || !fs::is_directory(input_directory)) {
		cerr << "Error: Input directory does not exist or is not a valid folder.\n";
		return 0;
	}

	// == Output Directory == //
	if (output_directory.empty()) {
		cout << "Please Enter The Desired output Path.: \n";
		getline(cin, output_directory);
		output_directory = clean_path(output_directory);
	}

	// Validating selected output directory (+ Creating it if it doesn't exist.)
	if (!fs::exists(output_directory)) {
		if (!fs::create_directory(output_directory)) {
			cerr << "Error: Could not create output directory!" << endl;
			return 0;
		}
	}
	return 1; // no error
}

bool load_images(vector<pair<string, Mat>>& input_images, const string& input_directory) {
	try {
		std::vector<fs::directory_entry> entries;

		for (const auto& entry : fs::directory_iterator(input_directory)) {
			if (entry.is_regular_file()) {
				entries.push_back(entry);
			}
		}

		std::sort(entries.begin(), entries.end(), naturalSort);

		for (const auto& entry : entries) {
			Mat img = imread(entry.path().string(), IMREAD_COLOR); // Load images in color
			if (img.empty()) {
				cerr << "Warning: Could not read image: " << entry.path() << endl;
				continue;
			}
			input_images.push_back({ entry.path().filename().string(), img });
			if (VERBOSE) {
				cout << "Loaded: " << entry.path() << endl;
			}
		}
	}
	catch (const fs::filesystem_error& e) {
		cerr << "Filesystem error: " << e.what() << endl;
		return 0;
	}

	if (input_images.empty()) {
		cerr << "Error: No images found in directory." << endl;
		return 0;
	}
	return 1; // no error
}

bool set_frames(int& nFrames_selected, const int& input_images_size) {
	if (VERBOSE) {
		cout << "How Many Frames would you like to process? (Min: 1, Max: " << input_images_size << "): ";
		cin >> nFrames_selected;
	}
	else {
		// For evaluation assume maximum input
		nFrames_selected = input_images_size;
	}

	if (nFrames_selected <= 0 || nFrames_selected > input_images_size)
	{
		cerr << "Error: Invalid number of frames selected. Exiting." << endl;
		return 0;
	}
	return 1;
}

bool validate_images_same_size(const vector<Mat>& images) {
	if (images.empty()) {
		cerr << "Error: No images to validate!" << endl;
		return false;
	}

	int standard_rows = images[0].rows;
	int standard_cols = images[0].cols;
	int standard_channels = images[0].channels(); // Check channels
#ifdef OMP
	int not_standard_sizes = 0;
#pragma omp parallel for
	for (int i = 1; i < images.size(); ++i) {
		if (images[i].rows != standard_rows || images[i].cols != standard_cols || images[i].channels() != standard_channels) {
			// OMP critical section for cerr might be needed if multiple threads hit this, but reduction handles the flag
			// For simplicity, error messages can be non-atomic here or printed post-loop if not_standard_sizes > 0.
			// This error message might print multiple times or interleave, better to just set flag.
						// cerr << "Error: Image at index " << i << " does not match properties (" ...
			not_standard_sizes++;
		}
	}

	if (not_standard_sizes)return false;
#endif // OMP
#ifndef OMP
	for (size_t i = 1; i < images.size(); ++i) {
		if (images[i].rows != standard_rows || images[i].cols != standard_cols || images[i].channels() != standard_channels) {
			cerr << "Error: Image at index " << i << " does not match properties ("
				<< standard_rows << "x" << standard_cols << "x" << standard_channels << "). "
				<< "Found (" << images[i].rows << "x" << images[i].cols << "x" << images[i].channels() << ")." << endl;
			return false;
		}
	}
#endif // !OMP


	if (VERBOSE) {
		cout << "All images have consistent properties: "
			<< standard_rows << "x" << standard_cols << "x" << standard_channels << "." << endl;
	}
	return true;

}

Mat compute_estimated_background_median_filter(const vector<Mat>& images) {

	if (images.empty()) {
		cerr << "Error: No images found!" << endl;
		return Mat(); // Return empty matrix
	}

	int num_images = images.size(); // Total number of images (M)
	// Get the size of images
	int rows = images[0].rows; // Number of rows (height)
	int cols = images[0].cols; // Number of columns (width)
	int channels = images[0].channels();

	if (channels != 3) {
		cerr << "Error: Median filter for color background expects 3-channel images." << endl;
		// Fallback or error handling can be added here
		// For now, let's attempt to process as is, might fail if not 3 channels
	}
	Mat background_median = Mat::zeros(rows, cols, CV_8UC3); // Output is color
#ifdef OMP
#pragma omp parallel for collapse(2)
#endif // OMP
	// Loop over all pixels 
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			vector<uchar> all_B_pixels, all_G_pixels, all_R_pixels;
			all_B_pixels.reserve(num_images);
			all_G_pixels.reserve(num_images);
			all_R_pixels.reserve(num_images);
			for (int img_idx = 0; img_idx < num_images; ++img_idx) {
				Vec3b pixel_bgr = images[img_idx].at<Vec3b>(row, col);
				all_B_pixels.push_back(pixel_bgr[0]);
				all_G_pixels.push_back(pixel_bgr[1]);
				all_R_pixels.push_back(pixel_bgr[2]);
			}

			uchar median_B = static_cast<uchar>(std::round(find_median(all_B_pixels)));
			uchar median_G = static_cast<uchar>(std::round(find_median(all_G_pixels)));
			uchar median_R = static_cast<uchar>(std::round(find_median(all_R_pixels)));

			background_median.at<Vec3b>(row, col) = Vec3b(median_B, median_G, median_R);
		}
	}
	return background_median; // Already CV_8UC3
}

Mat compute_estimated_background(const vector<Mat>& images) {

	if (images.empty()) {
		cerr << "Error: No images found!" << endl;
		return Mat(); // Return empty matrix
	}

	int num_images = images.size(); // Total number of images (M)
	// Get the size of images
	int rows = images[0].rows; // Number of rows (height)
	int cols = images[0].cols; // Number of columns (width)
	int channels = images[0].channels(); // Get number of channels

	if (channels != 3) {
		cerr << "Warning: compute_estimated_background processing non-3-channel images as color. Result might be unexpected." << endl;
	}
	// Create a matrix to accumulate sum of pixel values, We use floating point to avoid integer overflow
	Mat background_sum = Mat::zeros(rows, cols, CV_32FC3);// Sum for 3 channels


	// Loop over all images one by one
	for (int img_idx = 0; img_idx < num_images; ++img_idx) {

		// Get current image
		Mat current_image = images[img_idx];
		int r_idx = 0;
		int c_idx = 0;
#ifdef OMP
#pragma omp parallel for collapse(2) private(r_idx, c_idx)
#endif // OMP
		for (r_idx = 0; r_idx < rows; ++r_idx) {
			for (c_idx = 0; c_idx < cols; ++c_idx) {
				Vec3b pixel_value = current_image.at<Vec3b>(r_idx, c_idx);
				Vec3f pixel_value_float(static_cast<float>(pixel_value[0]),
					static_cast<float>(pixel_value[1]),
					static_cast<float>(pixel_value[2]));
				// This += on Vec3f is element-wise.
				// Since img_idx loop is sequential, there's no race on background_sum[r_idx,c_idx] for different images.
				// OMP parallelizes access to different (r_idx, c_idx) for the *same* current_image.
				background_sum.at<Vec3f>(r_idx, c_idx) += pixel_value_float;
			}
		}
	}

	//Create a matrix to store the final background (float type first)
	Mat background_mean = Mat::zeros(rows, cols, CV_32FC3); // Mean for 3 channels
	//float total_sum = 0;
	//float mean_value = 0;
	// Loop over each pixel again to divide sum by number of images (M)
	// B(row,col) = (pixel1(row,col) + pixel2(row,col) + ... + pixelM(row,col)) / M
	int r_idx = 0;
	int c_idx = 0;
#ifdef OMP
#pragma omp parallel for collapse(2) private(r_idx, c_idx)
#endif // OMP
	for (r_idx = 0; r_idx < rows; ++r_idx) {
		for (c_idx = 0; c_idx < cols; ++c_idx) {
			Vec3f total_sum_vec = background_sum.at<Vec3f>(r_idx, c_idx);
			Vec3f mean_value_vec;
			mean_value_vec[0] = total_sum_vec[0] / static_cast<float>(num_images);
			mean_value_vec[1] = total_sum_vec[1] / static_cast<float>(num_images);
			mean_value_vec[2] = total_sum_vec[2] / static_cast<float>(num_images);
			background_mean.at<Vec3f>(r_idx, c_idx) = mean_value_vec;
		}
	}

	// Convert the background from float to 8-bit for display/saving
	Mat background_final;
	background_mean.convertTo(background_final, CV_8UC3);

	//Return the computed background image
	return background_final;
}

bool extract_images_for_processing(vector<Mat>& images_only, const vector<pair<string, Mat>>& input_images, const int& nFrames_selected) {
	try {
		images_only.reserve(nFrames_selected);
		for (int i = 0; i < nFrames_selected; i++) {
			images_only.push_back(input_images[i].second);
		}
	}
	catch (const std::exception& e) {
		cerr << "Error: " << e.what() << endl;
		return 0;
	}
	return 1;
}

void save_display_background(const string& output_directory, const Mat& background) {
	imwrite(output_directory + "\\estimated_background.jpg", background);
	if (VERBOSE) {
		imshow("Estimated Background", background);
		waitKey(500); // Show for a short time
	}
}

Mat compute_foreground_mask(const Mat& background, const Mat& frame, int threshold_value) {
	if (background.empty() || background.type() != CV_8UC3) {
		cerr << "Error: Background image is empty or not 3-channel color!" << endl;
		return Mat();
	}
	if (frame.empty() || frame.type() != CV_8UC3) {
		cerr << "Error: Input frame image is empty or not 3-channel color!" << endl;
		return Mat();
	}

	Mat background_gray, frame_gray;
	cvtColor(background, background_gray, COLOR_BGR2GRAY);
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	int rows = background_gray.rows;
	int cols = background_gray.cols;

	Mat foreground_mask = Mat::zeros(rows, cols, CV_8UC1); // Mask is single channel
	int r_idx = 0;
	int c_idx = 0;
#ifdef OMP
#pragma omp parallel for collapse(2) private(r_idx, c_idx)
#endif // OMP
	for (r_idx = 0; r_idx < rows; ++r_idx) {
		for (c_idx = 0; c_idx < cols; ++c_idx) {
			int background_pixel_value = static_cast<int>(background_gray.at<uchar>(r_idx, c_idx));
			int frame_pixel_value = static_cast<int>(frame_gray.at<uchar>(r_idx, c_idx));
			int pixel_difference = abs(background_pixel_value - frame_pixel_value);

			if (pixel_difference > threshold_value) {
				foreground_mask.at<uchar>(r_idx, c_idx) = 255;
			}
			else {
				foreground_mask.at<uchar>(r_idx, c_idx) = 0;
			}
		}
	}
	return foreground_mask;
}

vector<Mat> generate_foreground_masks(const vector<Mat>& images_only, const Mat& background, int threshold_value = 59) {
	vector<Mat> masks(images_only.size());
#ifdef OMP
#pragma omp parallel for
#endif // OMP
	for (int i = 0; i < images_only.size(); i++) {
		const Mat& frame = images_only[i];
		Mat mask = compute_foreground_mask(background, frame, threshold_value);
		if (!mask.empty()) {
			masks[i] = mask;
		}
	}

	return masks;
}

void save_foreground_masks(const vector<Mat>& masks, const string& output_directory) {
	for (size_t i = 0; i < masks.size(); ++i) {
		string output_filename = output_directory + "\\foreground_mask_" + to_string(i) + ".jpg";
		imwrite(output_filename, masks[i]);
		if (VERBOSE) {
			cout << "Saved: " << output_filename << endl;

			imshow("Foreground Mask", masks[i]);
			waitKey(300);
		}
	}
	if (VERBOSE) {
		cout << "All foreground masks saved into " << output_directory << "!" << endl;
		system(("start \"\" \"" + output_directory + "\"").c_str());
	}
}



// === Algorithm Implementations === //


int sequential() {

#ifdef OMP

	// leave it commented to get the number of cores your device got
	// omp_set_num_threads(12);
#endif // OMP
	// === Receiving Directories === //
	string input_directory, output_directory;
	if (set_directories(input_directory, output_directory) == 0)
		return -1;


	vector<pair<string, Mat>> input_images;
	// === Loading Images ===
	if (load_images(input_images, input_directory) == 0)
		return -1;
	else if (VERBOSE)
		cout << "Successfully loaded " << input_images.size() << " images." << endl;


	// === Decide Number Of Frames To Be Processed === //
	int nFrames_selected = 0;
	if (set_frames(nFrames_selected, input_images.size()) == 0)
		return -1;

	vector<Mat> images_only;
	images_only.reserve(nFrames_selected);
	if (extract_images_for_processing(images_only, input_images, nFrames_selected) == 0) return -1;

	if (!validate_images_same_size(images_only)) {
		cerr << "Error: Images are not the same size/channels. Cannot continue." << endl;
		return -1;
	}

	auto start = chrono::high_resolution_clock::now();


	// === Compute Background Image ===
#ifdef MEDIAN_FILTER
	Mat background = compute_estimated_background_median_filter(images_only);
#endif
#ifndef MEDIAN_FILTER
	Mat background = compute_estimated_background(images_only);
#endif 

	if (background.empty()) {
		cerr << "Background computation failed!" << endl;
		return -1;
	}

	// === Compute Foreground Masks ===
	vector<Mat> masks = generate_foreground_masks(images_only, background);

	auto end = chrono::high_resolution_clock::now();

	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
	int seconds = duration.count() / 1000;
	int milliseconds = duration.count() % 1000;
	cout << "Execution time: " << seconds << ":" << milliseconds << " seconds" << endl;

	// === Save and Display the Estimated Background ===
	save_display_background(output_directory, background);

	// === Save and Display Foreground Masks ===
	save_foreground_masks(masks, output_directory);

	return 0;
}

int mpi() {
	int rank, size, chunk_size, padded_size;
	string input_directory, output_directory;
	vector<Mat> images_only_root; // Only root loads all images
	int rows = 0, cols = 0, channels = 0;
	int num_pixels_per_process = 0; // Number of pixels (not uchars) this process handles
	int num_elements_per_process = 0; // Number of uchars (pixels * channels) this process handles

	vector<Mat> images_only;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::chrono::steady_clock::time_point start, end;

	int nFrames_selected = 0;

	if (rank == 0) {
		// === Receiving Directories === //
		if (set_directories(input_directory, output_directory) == 0)
			MPI_Abort(MPI_COMM_WORLD, -1);

		vector<pair<string, Mat>> input_images;
		// === Loading Images ===
		if (load_images(input_images, input_directory) == 0)
			MPI_Abort(MPI_COMM_WORLD, -1);
		else if (VERBOSE)
			cout << "Successfully loaded " << input_images.size() << " images." << endl;

		// === Decide Number Of Frames To Be Processed ===
		if (set_frames(nFrames_selected, input_images.size()) == 0)
			MPI_Abort(MPI_COMM_WORLD, -1);

		// === Extract Images for Processing ===
		if (extract_images_for_processing(images_only, input_images, nFrames_selected) == 0)
			MPI_Abort(MPI_COMM_WORLD, -1);

		// === Validate All Images Have the Same Size ===
		if (!validate_images_same_size(images_only)) {
			cerr << "Error: Images are not the same size. Cannot continue." << endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		start = chrono::steady_clock::now();

		// === Flatten the images into 1d array ===
		rows = images_only[0].rows;
		cols = images_only[0].cols;

		channels = images_only[0].channels();

		int total_pixels_per_image = rows * cols;
		num_pixels_per_process = (total_pixels_per_image + size - 1) / size;
		num_elements_per_process = num_pixels_per_process * channels;


	}

	// === Broadcast images information needed in worker nodes
	MPI_Bcast(&nFrames_selected, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_pixels_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&num_elements_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

	vector<vector<uchar>> local_flattened_image_chunks(nFrames_selected);
	for (int i = 0; i < nFrames_selected; ++i) {
		vector<uchar> image_data_for_scatter_root;
		if (rank == 0) {
			const Mat& img = images_only_root[i];
			image_data_for_scatter_root.assign(img.datastart, img.dataend);
			// Pad if necessary for scatter: total elements for scatter should be num_elements_per_process * size
			image_data_for_scatter_root.resize(num_elements_per_process * size, 0);
		}

		local_flattened_image_chunks[i].resize(num_elements_per_process);
		MPI_Scatter(image_data_for_scatter_root.data(), num_elements_per_process, MPI_UNSIGNED_CHAR,
			local_flattened_image_chunks[i].data(), num_elements_per_process, MPI_UNSIGNED_CHAR,
			0, MPI_COMM_WORLD);
	}

	vector<Mat> local_image_strips(nFrames_selected);
	for (int i = 0; i < nFrames_selected; ++i) {
		// Create a Mat view, then clone. Data for Mat is num_pixels_per_process * channels = num_elements_per_process
		local_image_strips[i] = Mat(1, num_pixels_per_process, CV_MAKETYPE(CV_8U, channels), local_flattened_image_chunks[i].data()).clone();
	}

	// === Compute Background Image ===
#ifdef MEDIAN_FILTER
	Mat background = compute_estimated_background_median_filter(local_image_strips);
#endif
#ifndef MEDIAN_FILTER
	Mat background = compute_estimated_background(local_image_strips);
#endif 
	if (background.empty()) {
		cerr << "Background computation failed!" << endl;
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	// === Compute Foreground Masks ===
	vector<Mat> masks = generate_foreground_masks(local_image_strips, background);

	// === Flatten the background and foreground masks into 1D arrays ===
	vector<uchar> flattened_background;
	flattened_background.assign(background.datastart, background.dataend);

	vector<vector<uchar>> flattened_masks(nFrames_selected);
	for (int i = 0; i < nFrames_selected; ++i) {
		// Flatten the mask into a 1D array
		flattened_masks[i].assign(masks[i].data, masks[i].data + masks[i].total());
	}

	// === Gather the background and foreground masks back to process 0 ===
	vector<uchar> gathered_background;
	if (rank == 0) {
		gathered_background.resize(rows * cols);
	}
	MPI_Gather(flattened_background.data(), num_elements_per_process, MPI_UNSIGNED_CHAR,
		gathered_background.data(), num_elements_per_process, MPI_UNSIGNED_CHAR,
		0, MPI_COMM_WORLD);

	vector<vector<uchar>> gathered_masks(nFrames_selected);
	for (int i = 0; i < nFrames_selected; ++i) {
		if (rank == 0) {
			gathered_masks[i].resize(num_pixels_per_process * size);
		}
		MPI_Gather(flattened_masks[i].data(), num_pixels_per_process, MPI_UNSIGNED_CHAR,
			gathered_masks[i].data(), num_pixels_per_process, MPI_UNSIGNED_CHAR,
			0, MPI_COMM_WORLD);
	}

	if (rank == 0) {
		// Resize to actual image size to remove padding
		gathered_background.resize(rows * cols * channels);
		for (int i = 0; i < nFrames_selected; ++i) {
			gathered_masks[i].resize(rows * cols);
		}
		Mat final_background(rows, cols, CV_MAKETYPE(CV_8U, channels), gathered_background.data());
		Mat final_background_owned = final_background.clone();
		vector<Mat> final_masks(nFrames_selected);


		for (int i = 0; i < nFrames_selected; ++i) {
			final_masks[i] = Mat(rows, cols, CV_8UC1, gathered_masks[i].data()).clone();
		}

		end = chrono::steady_clock::now();

		auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
		int seconds = duration.count() / 1000;
		int milliseconds = duration.count() % 1000;
		cout << "Execution time: " << seconds << ":" << milliseconds << " seconds" << endl;

		// === Save and Display the Estimated Background ===
		save_display_background(output_directory, final_background);

		// === Save and Display Foreground Masks ===
		save_foreground_masks(final_masks, output_directory);
	}

	MPI_Finalize();
	return 0;
}


// === Main Function === //


int main() {
	int (*mode_function_ptr)() = nullptr;

	if (string(MODE) == "sequential") {
		mode_function_ptr = sequential;
	}
	else if (string(MODE) == "mpi") {
		mode_function_ptr = mpi;
	}
	else {
		cerr << "Error: Unknown mode." << endl;
		return 1;
	}

	int result = mode_function_ptr();

	return result;
}
