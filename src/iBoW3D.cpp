#include "iBoW3D.h"

using namespace cv;
using namespace std;

namespace iBoW3D
{
    iBoW3D::iBoW3D(std::shared_ptr<geometry::PointCloud> current_pcd_, 
                   int frameID_, int init_words_num_, int words_num_add_, 
                   int keypoint_num_, int feature_dim_, 
                   const std::string key_feature_path_, const std::string all_feature_path_, 
                   DataBase *pDB_,
                   double lambda_word_, int near_num_, int search_num_, 
                   double score_th_, double fit_th_, double check_th_, double score_th_2_, double fit_th_2_,
                   bool prior_fit_, int gap_num_,
                   int max_iter_, int ransac_n_, bool remove_outliers_):
    frameID(frameID_), lastLoopID(-100), LoopID(-1),
    current_pcd(current_pcd_),
    key_feature_path(key_feature_path_), all_feature_path(all_feature_path_), 
    pDB(pDB_),
    words_num(init_words_num_), words_num_add(words_num_add_),
    keypoint_num(keypoint_num_), feature_dim(feature_dim_), 
    lambda_word(lambda_word_), near_num(near_num_), search_num(search_num_), 
    score_th(score_th_), fit_th(fit_th_), check_th(check_th_), score_th_2(score_th_2_), fit_th_2(fit_th_2_),
    prior_fit(prior_fit_), gap_num(gap_num_),
    max_iter(max_iter_), ransac_n(ransac_n_), distance_length(100), remove_outliers(remove_outliers_)
    {
        // current frame info
        pcd_num = current_pcd->points_.size();
        pFC = new FeatureContainer(keypoint_num, feature_dim, pcd_num, frameID, key_feature_path, all_feature_path);

        /* current_hist = Mat::zeros(Size(1,words_num), CV_16S); */
        current_hist.resize(words_num,1);
        current_hist.setZero();

        idf = Eigen::ArrayXd::Zero(words_num);

        trans = Eigen::Matrix4d_u::Identity();

    }

    void iBoW3D::update_current_frame(std::shared_ptr<geometry::PointCloud> current_pcd_, int frameID_)
    {
        current_pcd = current_pcd_;
        frameID = frameID_;
        pcd_num = current_pcd->points_.size();
        pFC = new FeatureContainer(keypoint_num, feature_dim, pcd_num, frameID, key_feature_path, all_feature_path);
    }

    void iBoW3D::get_dictionary_and_histogram()
    {
        // get dictionary
        vector<cv::Mat> key_feat_DB = pDB->get_key_feat();
        int DB_len = pDB->get_DB_len();
        int key_feat_total_num = keypoint_num * DB_len;
        Mat keyFeatures(key_feat_total_num, feature_dim, CV_32F, Scalar(0));
        Mat temp; 
        for(int i = 0; i < DB_len; i++)
        {
            temp = keyFeatures(Rect(0, i*keypoint_num, feature_dim, keypoint_num));
            key_feat_DB[i].copyTo(temp);
        }

        // implement k-means with k-means++ seeds
        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1);
        kmeans(keyFeatures, words_num, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

        // calculate idf
        Eigen::ArrayXd idf_temp = Eigen::ArrayXd::Zero(words_num);
        idf = Eigen::ArrayXd::Zero(words_num); // set every element of idf = 0
        

        for(int i = 0; i < key_feat_total_num; i++)
        {
            idf_temp(labels.at<int>(i))++;
        }
        for(int i = 0; i < words_num; i++)
        {
            idf(i) = log(key_feat_total_num/(idf_temp(i)+1));
        }

        // get histograms
        hist_DB.clear();
        for(int i = 0; i < DB_len; i++)
        {
            
            Eigen::VectorXd hist_temp = Eigen::VectorXd::Zero(words_num);

            for(int j = 0; j < keypoint_num; j++)
            {
                hist_temp(labels.at<int>(i*keypoint_num+j)) += 1.0;
            }


            // get tf: hist_temp is divided by keypoint_num which is equal the number of features of current scan
            hist_temp = hist_temp / keypoint_num;
            // then multiple idf
            hist_temp = (hist_temp.array()*idf).matrix();
            // l2 normalize
            hist_temp.normalize();
            // add to database
            hist_DB.push_back(hist_temp);
        }
    }


    void iBoW3D::update_dictionary_histograms()
    {
        cout << "------------------ Update Dictionary -----------------" << endl;
        words_num += words_num_add;
        get_dictionary_and_histogram();
    }


    void iBoW3D::get_current_histogram()
    {

        current_hist.resize(words_num,1);
        current_hist.setZero();

        Mat temp(1, feature_dim, CV_32F);

        vector<double> dist_list; // save distances
        int min_idx = 0;

        

        for(int i = 0; i < keypoint_num; i++)
        {
            dist_list.clear();
            temp = (pFC->getKeyFeature()).row(i);
            // identify the min index e.g. label
            for(int j = 0; j < words_num; j++)
            {
                double dist = norm(temp-centers.row(j), NORM_L2);
                dist_list.push_back(dist);
            }
            min_idx = distance(dist_list.begin(), min_element(dist_list.begin(), dist_list.end()));
            // corresponding number of word plus 1
            current_hist(min_idx) += 1.0;
        }


        // get tf: current_hist is divided by keypoint_num which is equal the number of features of current scan
        current_hist = current_hist / keypoint_num;
        // then multiple idf
        current_hist = (current_hist.array()*idf).matrix();
        // l2 normalize
        current_hist.normalize();
        
    }


    void iBoW3D::update_database()
    {
        pDB->add_pcloud(current_pcd);
        pDB->add_key_feat(pFC->getKeyFeature());
        pDB->add_all_feat(pFC->getAllFeature());
        pDB->add_frameIdx(frameID);
        pDB->add_DBIdx(frameID, 1); // DB_len has not added one
        pDB->add_DB_len();
    }

    void iBoW3D::update_only_DBIdx()
    {
        pDB->add_DBIdx(frameID, 0);
    }


    int iBoW3D::retrieve()
    {
        vector<vector<int>> cand_list = cand_selection();

        // if no loop, LoopID = -1
        if (cand_list.empty())
        {
            return -1;
        }
        else
        {
            vector<vector<int>> cand_list_2 = range_check(cand_list);

            if (cand_list_2.empty())
            {
                return -1;
            }
            else
            {
                cout << "Candidate List: " << endl;
                vector<vector<int>>::iterator iter;
                for(iter=cand_list_2.begin(); iter!=cand_list_2.end(); iter++)
                {
                    for(int i=0; i<(int)((*iter).size()); i++)
                    {
                        cout << (*iter)[i] << " ";
                    }
                    cout << endl;
                }
                cout << endl;

                LoopID = geo_verification_ransac(cand_list_2);

                return LoopID;
            }
        }
        
    }

    vector<vector<int>> iBoW3D::cand_selection()
    {
        
        // choose the non-zero part of current histogram
        vector<int> chosen_idx_non_zero;
        int non_zero_num;
        int i = 0;
        for(i=0; i < words_num; i++)
        {
            if (current_hist(i)>0){
                chosen_idx_non_zero.push_back(i);
            }
        }
        non_zero_num = (int)(chosen_idx_non_zero.size());
        
        /* hist_cand saves all candidates; 
           hist_idx_cand is the real frame idx, hist_idx_cand_DB is the idx of Database */
        vector<int> hist_idx_cand;
        vector<int> hist_idx_cand_DB;

        // coarse selection
        vector<Eigen::VectorXd>::iterator iter;
        Eigen::VectorXd temp_hist;
        vector<int> frameIdx = pDB->get_frameIdx();
        i = 0;
        for(iter=hist_DB.begin(); iter!=hist_DB.end(); iter++)
        {
            if(frameIdx[i] > frameID-gap_num)
            {
                break;
            }

            temp_hist = (*iter)(chosen_idx_non_zero, Eigen::all);

            if ( (temp_hist.array() > 0).select(1, temp_hist).sum() > lambda_word*non_zero_num ){
                hist_idx_cand.push_back(frameIdx[i]);
                hist_idx_cand_DB.push_back(i);
            }
            i++;
        }
        if (hist_idx_cand.size()==0) // no scans after coarse selection
        {
            vector<vector<int>> island;
            return island;
        }
        else
        {
            // fine selection
            // calculate distances and save
            vector<int>::iterator iter;
            vector<pair<double, int>> distance_list;
            double dist1, dist2, dist;
            for(iter=hist_idx_cand_DB.begin(); iter!=hist_idx_cand_DB.end(); iter++)
            {
                dist1 = (current_hist-hist_DB[*iter]).norm();
                dist2 = current_hist.dot(hist_DB[*iter]);
                dist = dist1;
                distance_list.push_back(make_pair(dist, hist_idx_cand[iter-hist_idx_cand_DB.begin()]));
            }
            // sort distances
            sort(distance_list.begin(),distance_list.end(), [](const pair<double, int>& a, const pair<double, int>& b){ return a.first < b.first; });
            
            vector<pair<double, int>>::iterator iter1, iter2;
            vector<vector<int>> idx_island_list;
            vector<double> avg_dist_list;
            vector<int> idx_num_list;
            vector<int> idx_island_temp;
            int dist_sum = 0;
            int idx_num = 0;
            int chosen_old_loop_flag = 0;
            vector<int> island1, island2, island3;
            for(iter1=distance_list.begin(); iter1!=distance_list.end(); iter1++)
            {
                idx_island_temp.clear();
                dist_sum = 0;
                idx_num = 0;
                for(iter2=distance_list.begin(); iter2!=distance_list.end(); iter2++)
                {
                    if (abs((*iter2).second - (*iter1).second) < near_num){
                        idx_island_temp.push_back((*iter2).second);
                        dist_sum += (*iter2).first;
                        idx_num++;
                    }
                }
                idx_island_list.push_back(idx_island_temp);
                avg_dist_list.push_back(dist_sum/idx_num);
                idx_num_list.push_back(idx_num);
                if (chosen_old_loop_flag==0 && abs(lastLoopID-(*iter1).second)<near_num/2)
                {
                    island3.assign(idx_island_temp.begin(), idx_island_temp.begin() + min({idx_num, search_num}));
                    chosen_old_loop_flag = 1;
                }
            }
            int max_num_idx = distance(idx_num_list.begin(), max_element(idx_num_list.begin(), idx_num_list.end()));
            island1.assign(idx_island_list[max_num_idx].begin(), idx_island_list[max_num_idx].begin() + min({idx_num_list[max_num_idx], search_num}));

            int min_dist_idx = distance(avg_dist_list.begin(), min_element(avg_dist_list.begin(), avg_dist_list.end()));
            island2.assign(idx_island_list[min_dist_idx].begin(), idx_island_list[min_dist_idx].begin() + min({idx_num_list[min_dist_idx], search_num}));

            vector<int> island22, island32;
            vector<int>::iterator find_it, find_it_2;
            for(int i=0; i<(int)(island2.size()); i++)
            {
                find_it = find(island1.begin(), island1.end(), island2[i]);
                if(find_it==island1.end()) { island22.push_back(island2[i]); }
            }
            if (chosen_old_loop_flag == 1) // there is island3
            {
                for(int i=0; i<(int)(island3.size()); i++)
                {
                    find_it = find(island1.begin(), island1.end(), island3[i]);
                    if(find_it==island1.end())
                    {
                        find_it_2 = find(island22.begin(), island22.end(), island3[i]);
                        if(find_it_2==island22.end()) { island32.push_back(island3[i]); }
                    }
                }
            }

            vector<vector<int>> cand_list;

            sort(island1.begin(),island1.end(), [](const int& a, const int& b){ return a < b; });
            if(island1.size() > 2)
            {
                swap(island1[0], island1[(int)(island1.size())/2]);
            }
            sort(island22.begin(),island22.end(), [](const int& a, const int& b){ return a < b; });
            if(island22.size() > 2)
            {
                swap(island22[0], island22[(int)(island22.size())/2]);
            }

            cand_list.push_back(island1);
            cand_list.push_back(island22);

            if (chosen_old_loop_flag == 1) // there is island3
            {
                sort(island32.begin(),island32.end(), [](const int& a, const int& b){ return a < b; });
                if(island32.size() > 2)
                {
                    swap(island32[0], island32[(int)(island32.size())/2]);
                }
                cand_list.push_back(island32);
            }

            return cand_list;
        }
        
    }


    void iBoW3D::pcd_ransac(pair<int, int> cand_pair, vector<result_struct> & res_list,
                            std::shared_ptr<geometry::PointCloud> current_pcd,
                            std::shared_ptr<pipelines::registration::Feature> current_features)
    {
        int iter_find = pDB->get_DBIdx_by_idx(cand_pair.second);

        std::shared_ptr<geometry::PointCloud> pCand = pDB->get_cloud_by_idx(iter_find);
        cv::Mat cand_all_feat = pDB->get_all_feat_by_idx(iter_find);

        std::shared_ptr<pipelines::registration::Feature> cand_features(new pipelines::registration::Feature);
        int cand_size = int(pCand->points_.size());
        cand_features->Resize(feature_dim, cand_size);
        Eigen::MatrixXf cand_data_f(feature_dim, cand_size);
        Eigen::MatrixXd cand_data(feature_dim, cand_size);

        cv::cv2eigen(cand_all_feat, cand_data_f);
        cand_data = cand_data_f.cast<double>();
        cand_features->data_ = cand_data.transpose();

        pipelines::registration::RegistrationResult registration_result;

        bool mutual_filter = false;
        double distance_threshold = 5.0;
        int max_iteration = max_iter;

        // Prepare checkers
        std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
        auto correspondence_checker_edge_length = pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
        auto correspondence_checker_distance = pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);
        correspondence_checker.push_back(correspondence_checker_edge_length);
        correspondence_checker.push_back(correspondence_checker_distance);

        open3d::utility::random::Seed(1);
        registration_result = pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
                *current_pcd, *pCand, *current_features, *cand_features,
                mutual_filter, distance_threshold,
                pipelines::registration::TransformationEstimationPointToPoint(false), 
                ransac_n, correspondence_checker,
                pipelines::registration::RANSACConvergenceCriteria(max_iteration, 0.999));

        res_list[cand_pair.first].fit = registration_result.fitness_;
        res_list[cand_pair.first].score = registration_result.inlier_rmse_;
        res_list[cand_pair.first].trans = registration_result.transformation_;
        res_list[cand_pair.first].idx = cand_pair.second;
    }


    void iBoW3D::island_ransac(pair<int, vector<int>> island_pair, vector<result_struct> & res_list,
                               std::shared_ptr<geometry::PointCloud> current_pcd,
                               std::shared_ptr<pipelines::registration::Feature> current_features)
    {
        int iter_find = pDB->get_DBIdx_by_idx((island_pair.second)[0]);

        std::shared_ptr<geometry::PointCloud> pCand;
        std::shared_ptr<pipelines::registration::Feature> cand_features(new pipelines::registration::Feature);

        pCand = pDB->get_cloud_by_idx(iter_find);
        cv::Mat cand_all_feat = pDB->get_all_feat_by_idx(iter_find);

        int cand_size = int(pCand->points_.size());
        cand_features->Resize(feature_dim, cand_size);
        Eigen::MatrixXf cand_data_f(feature_dim, cand_size);
        Eigen::MatrixXd cand_data(feature_dim, cand_size);

        cv::cv2eigen(cand_all_feat, cand_data_f);
        cand_data = cand_data_f.cast<double>();
        cand_features->data_ = cand_data.transpose();

        // Perform alignment
        pipelines::registration::RegistrationResult registration_result;

        bool mutual_filter = false;
        double distance_threshold = 5.0;
        int max_iteration = max_iter;

        // Prepare checkers
        std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
        auto correspondence_checker_edge_length = pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
        auto correspondence_checker_distance = pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);
        correspondence_checker.push_back(correspondence_checker_edge_length);
        correspondence_checker.push_back(correspondence_checker_distance);

        open3d::utility::random::Seed(1);
        registration_result = pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
                *current_pcd, *pCand, *current_features, *cand_features,
                mutual_filter, distance_threshold,
                pipelines::registration::TransformationEstimationPointToPoint(false), 
                ransac_n, correspondence_checker,
                pipelines::registration::RANSACConvergenceCriteria(max_iteration, 0.999));

        if(registration_result.fitness_>fit_th_2 && registration_result.inlier_rmse_<score_th_2)
        {
            int current_res_list_idx = island_pair.first;
            res_list[current_res_list_idx].fit = registration_result.fitness_;
            res_list[current_res_list_idx].score = registration_result.inlier_rmse_;
            res_list[current_res_list_idx].trans = registration_result.transformation_;
            res_list[current_res_list_idx].idx = (island_pair.second)[0];

            vector<pair<int, int>> island_with_idx_list;
            for(int i=1; i<(int)((island_pair.second).size()); i++)
            {
                island_with_idx_list.push_back(make_pair(current_res_list_idx+i, (island_pair.second)[i]));
            }

            std::for_each(std::execution::par, island_with_idx_list.begin(), island_with_idx_list.end(), 
                        [this, &res_list, current_pcd, current_features](pair<int, int> cand_pair)
                        { pcd_ransac(cand_pair, res_list, current_pcd, current_features); });

        }
    }

    int iBoW3D::geo_verification_ransac(vector<vector<int>> cand_list)
    {
        std::shared_ptr<pipelines::registration::Feature> current_features(new pipelines::registration::Feature);        
        int current_size = int(current_pcd->points_.size());
        current_features->Resize(feature_dim, current_size);
        Eigen::MatrixXf current_data_f(feature_dim, current_size);
        Eigen::MatrixXd current_data(feature_dim, current_size);
 
        cv::Mat curr_all_feat = pFC->getAllFeature();

        cv::cv2eigen(curr_all_feat, current_data_f);
        current_data = current_data_f.cast<double>();
        current_features->data_ = current_data.transpose();

        vector<result_struct> res_list;
        vector<pair<int, vector<int>>> new_cand_list;

        int cand_list_len = 0;
        for(int i=0; i<(int)(cand_list.size()); i++)
        {
            new_cand_list.push_back(make_pair(cand_list_len, cand_list[i]));
            cand_list_len += (int)(cand_list[i].size());
        }
        for(int i=0; i<cand_list_len; i++)
        {
            result_struct res;
            res_list.push_back(res);
        }

        std::for_each(std::execution::par, new_cand_list.begin(), new_cand_list.end(), 
                        [this, &res_list, current_features](pair<int, vector<int>> island_pair){ island_ransac(island_pair, res_list, this->current_pcd, current_features); });
        
        cout << "res_list: " << res_list.size() << endl;
        for(int i=0; i<(int)(res_list.size()); i++)
        {
            cout << "fitness: " << res_list[i].fit << " ";
            cout << "RMSE: " << res_list[i].score << " ";
            cout << "idx: " << res_list[i].idx << endl;
        }

        if (prior_fit)
        {
            // get the idx of highest fit, more fit is better; sort with fit in descending order

            sort(res_list.begin(),res_list.end(), [](const result_struct& a, const result_struct& b){ return a.fit > b.fit; });
            for(int i=0; i<(int)(res_list.size()); i++)
            {
                if( res_list[i].fit>fit_th && res_list[i].score<score_th )
                {
                    cout << "Fit:" << res_list[i].fit << endl;
                    cout << "Score:" << res_list[i].score << endl;
                    Eigen::Matrix4d_u transformation = res_list[i].trans;
                    trans = transformation;
                    lastLoopID = res_list[i].idx;
                    return lastLoopID;
                }
                else
                {
                    continue;
                }
            }
            return -1;
        }
        else
        {
            // get the idx of least score, less score is better; sort with score in ascending order

            sort(res_list.begin(),res_list.end(), [](const result_struct& a, const result_struct& b){ return a.score < b.score; });
            for(int i=0; i<(int)(res_list.size()); i++)
            {
                if( res_list[i].fit>fit_th && res_list[i].score<score_th )
                {
                    cout << "Fit:" << res_list[i].fit << endl;
                    cout << "Score:" << res_list[i].score << endl;
                    Eigen::Matrix4d_u transformation = res_list[i].trans;
                    trans = transformation;
                    lastLoopID = res_list[i].idx;
                    return lastLoopID;
                }
                else
                {
                    continue;
                }
            }
            return -1;
        }
        
    }

    vector<vector<int>> iBoW3D::range_check(vector<vector<int>> cand_list)
    {

        double max_x_curr, min_x_curr, max_y_curr, min_y_curr;
        vector<Eigen::Vector3d> current_pcd_points = current_pcd->points_;
        sort(current_pcd_points.begin(),current_pcd_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[0] < b[0]; });
        max_x_curr = current_pcd_points[(int)(current_pcd_points.size()*0.995)][0];
        min_x_curr = current_pcd_points[(int)(current_pcd_points.size()*0.005)][0];
        sort(current_pcd_points.begin(),current_pcd_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[1] < b[1]; });
        max_y_curr = current_pcd_points[(int)(current_pcd_points.size()*0.995)][1];
        min_y_curr = current_pcd_points[(int)(current_pcd_points.size()*0.005)][1];


        std::shared_ptr<geometry::PointCloud> pCand;
        double max_x_cand, min_x_cand, max_y_cand, min_y_cand;
        double max_x_rate, min_x_rate, max_y_rate, min_y_rate;
        
        vector<vector<int>> new_cand_list;

        Eigen::Vector3d max_point_2, min_point_2;
        vector<vector<int>>::iterator iter;
        for(iter=cand_list.begin(); iter!=cand_list.end(); iter++)
        {
            vector<int> new_island;
            for(int i=0; i<(int)((*iter).size()); i++)
            {
                int iter_find = pDB->get_DBIdx_by_idx((*iter)[i]);
                pCand = pDB->get_cloud_by_idx(iter_find);

                vector<Eigen::Vector3d> pCand_points = pCand->points_;
                sort(pCand_points.begin(),pCand_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[0] < b[0]; });
                max_x_cand = pCand_points[(int)(pCand_points.size()*0.995)][0];
                min_x_cand = pCand_points[(int)(pCand_points.size()*0.005)][0];
                sort(pCand_points.begin(),pCand_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[1] < b[1]; });
                max_y_cand = pCand_points[(int)(pCand_points.size()*0.995)][1];
                min_y_cand = pCand_points[(int)(pCand_points.size()*0.005)][1];

                max_x_rate = max_x_cand/max_x_curr;
                min_x_rate = min_x_cand/min_x_curr;
                max_y_rate = max_y_cand/max_y_curr;
                min_y_rate = min_y_cand/min_y_curr;

                if(1/check_th < max_x_rate && max_x_rate < check_th &&
                   1/check_th < min_x_rate && min_x_rate < check_th &&
                   1/check_th < max_y_rate && max_y_rate < check_th &&
                   1/check_th < min_y_rate && min_y_rate < check_th)
                {
                    new_island.push_back((*iter)[i]);
                }
            }
            if(!new_island.empty()) // new_island is not empty
            {
                new_cand_list.push_back(new_island);
            }
        }

        return new_cand_list;
    }


}


