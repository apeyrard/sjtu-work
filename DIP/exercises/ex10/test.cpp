#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
// Normalizes images in range between 0 and 255.
Mat normalize(const Mat& src) {
    Mat srcnorm;
    normalize(src, srcnorm, 0, 255, NORM_MINMAX, CV_8UC1);
    return srcnorm;
}


int main(int argc, const char *argv[]) {
    // Holds the images:
    cout << "test" <<endl;
    waitKey(0);
    vector<Mat> db;

    db.push_back(imread("washingtonDC_Band1.tif", 0));
    db.push_back(imread("washingtonDC_Band2.tif", 0));
    db.push_back(imread("washingtonDC_Band3.tif", 0));
    db.push_back(imread("washingtonDC_Band4.tif", 0));
    db.push_back(imread("washingtonDC_Band5.tif", 0));
    db.push_back(imread("washingtonDC_Band6.tif", 0));

    // create a matrix with the data in row:
    int total = db[0].rows * db[0].cols;
        Mat mat(total, db.size(), CV_32FC1);

        for(int i = 0; i < db.size(); i++) {
        Mat X = mat.col(i);
        db[i].reshape(1, total).col(0).convertTo(X, CV_32FC1, 1/255.);
    }
    // Number of components to keep for the PCA:
    int num_components = 3;

    // Perform a PCA:
    PCA pca(mat, Mat(), CV_PCA_DATA_AS_COL, num_components);

    // The mean face:
    imshow("avg", pca.mean.reshape(1, db[0].rows));

    // The first three eigenfaces:
    imshow("pc1", normalize(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", normalize(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", normalize(pca.eigenvectors.row(2)).reshape(1, db[0].rows));

            waitKey(0);


   return 0;
}
