%% This validation file is for testing the estimators and then exporting the outputs in a specific way.

%% name of the simulation
simulation = "estimator_designs_main";

%% measurements that I want to use for testing the estimators
test_measurements = [8 16 31 37 43 48];

%% locations
meas_loc = "C:\work\0_currently_in_use\AI_estimator_dev_toolchain\1_data_extraction_for_training\2_extracted_measurements\1_extracted_measurements_mat\ref_1_100Hz_2\ref_";
meas_ai_loc = "C:\work\0_currently_in_use\AI_estimator_dev_toolchain\1_data_extraction_for_training\2_extracted_measurements\1_extracted_measurements_mat\ref_1_100Hz_norm\ref_";
results_loc = "C:\work\AI_training\2_trained_models\other_estimators\results\lon\";

%% genearal parameters
R = 0.3615; % m (effective wheel radius)
m = 3020; % kg
theta_z = 7145; % kg*m^2
front_track_width = 1.662; % m
rear_track_width = 1.684; % m
wheelbase = 3.215; % m
COG_position_ratio = 0.521;

wheelbase_COG_front = wheelbase * COG_position_ratio; % m
wheelbase_COG_rear = wheelbase * (1-COG_position_ratio); % m

% 2D position vectors (CoG -> wheels)
r_COG_FL = [wheelbase_COG_front; front_track_width / 2; 0]; % m
r_COG_FR = [wheelbase_COG_front; -front_track_width / 2; 0]; % m
r_COG_RL = [-wheelbase_COG_rear; rear_track_width / 2; 0]; % m
r_COG_RR = [-wheelbase_COG_rear; -rear_track_width / 2; 0]; % m

% yaw_rate_ini = [0; 0; 1]; 
% veh_ini = [1; 0; 0]; 

% friction coefficents
c_lon_f = 180000; % -
c_lon_r = 200000; % -
c_lat_f = 140000; % -
c_lat_r = 150000; % -

% epsilons for saturation at the different models
eps_m_v = 1; % -
eps_bw_acc = 0.1; % -

dt = 0.01; % s

% for the torque based model
brk_coef_f = 19; % -
brk_coef_r = 20; % -

% for the ekf
x0 = [0.0001; 0.0001; 0.0001]; 
P0 = diag([1e-5, 1e-5, 1e-5]);
Q_ekf_1 = diag([1e-5, 1e-2, 1e-2]);
R_ekf_1 = 1e-2;

eps_ekf_m_bw_v_u = 0.5; % [m/s]
thr_ekf_m_bw_v_u = 5; % [m/s]

% Load ONNX model
ort = py.importlib.import_module('onnxruntime');
sess = ort.InferenceSession("C:\work\AI_training\2_trained_models\TCN\ref\it_1_norm\traced_models\lon\model_TCN_lon_129_traced.onnx");

np = py.importlib.import_module('numpy');

sequence_length = 50;
num_of_signals = 12;

%% running the simulation

for meas_num = test_measurements
    meas = load(meas_loc + meas_num() + ".mat");
    meas_norm = load(meas_ai_loc + meas_num + "_norm.mat");

    len = length(meas.time_table_100.imu_COG_acc_x);

    T = (len - 1) * 0.01;

    out = sim(simulation);

    v_u_best_wheel = out.best_wheel;
    v_u_model_based = out.model_based;
    v_u_ekf_m_bw = out.ekf_m_bw;
    v_u_veh_ref = out.veh_ref;
    veh_u = out.veh_u;
    veh_v = out.veh_v;
    
    v_u_ekf_ai = zeros(len, 1);
    for i = 1:len
        v_u_ekf_ai(i) = out.ekf_ai.Data(:,:,i);
    end

    time = seconds(v_u_best_wheel.Time);

    TT = timetable(time, veh_u.Data, veh_v.Data, v_u_best_wheel.Data, v_u_model_based.Data, v_u_ekf_m_bw.Data, v_u_veh_ref.Data, v_u_ekf_ai, ...
        'VariableNames', {'veh_u', 'veh_v', 'best_wheel','model_based','ekf_m_bw', 'veh_ref', 'ekf_ai'});

    T = timetable2table(TT);

    writetable(T, results_loc + "ref_" + meas_num + ".csv", "Delimiter", ','); 
end

