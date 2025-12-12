%--------------------------------------%
% Overview of the main toolchain code: %
%--------------------------------------%

% 1. Extracts the simulation data from the measurements (CAN signals,
%    reference GPS speeds) for training (*.csv) and testing (*.mat -
%    timetable). These measurement files where created with MPF, that 


% It is better to hit a clear function after every part of the code

%% 1. a) Setting the locations for extracting

% The location of the measurement file (without the numbering)
location_read_og = "C:\work\0_currently_in_use\AI_estimator_dev_toolchain\1_data_extraction_for_training\2_extracted_measurements\1_extracted_measurements_mat\ref_1_500Hz_2\";

% The location of the created .csv/.mat files
location_write_og = "C:\work\0_currently_in_use\AI_estimator_dev_toolchain\1_data_extraction_for_training\2_extracted_measurements\1_extracted_measurements_mat\ref_1_100Hz_2\";

% the base name of the .csv/.mat (timetable) file
file_name_og = ref_";

% choose the measurement ID which will be extracted
meas_number_list = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 22 23 24 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49];

% meas_number_list = 43;

%% 1. b) checking the min and max values across the measurements

% Normalizing the data - going through all the measurements and taking out
% the min and max values for all the signals
min_values = inf(1,20);
max_values = -inf(1,20);

min_max_table = array2table([min_values; max_values], 'VariableNames', {'imu_COG_acc_x', 'imu_COG_acc_y', 'imu_COG_acc_z',...
        'imu_COG_gyro_roll_rate', 'imu_COG_gyro_pitch_rate', 'imu_COG_gyro_yaw_rate',...
        'wheel_speed_FL', 'wheel_speed_FR', 'wheel_speed_RL', 'wheel_speed_RR',...
        'drive_torque_FL', 'drive_torque_FR', 'drive_torque_RL', 'drive_torque_RR',...
        'brake_pressure_FL', 'brake_pressure_FR', 'brake_pressure_RL', 'brake_pressure_RR',...
        'rwa_FM', 'rwa_RM'});

% itt csak ki kell szedni az összes signal közül a min és max értékeket

for signal_it = 1:20
    for meas_it = 1:length(meas_number_list)
        load(location_read_og + file_name_og + int2str(meas_number_list(meas_it)) + ".mat");
        
        min_tmp = min(time_table_100.(min_max_table.Properties.VariableNames{signal_it}));
        max_tmp = max(time_table_100.(min_max_table.Properties.VariableNames{signal_it}));

        if(min_tmp < min_max_table.(min_max_table.Properties.VariableNames{signal_it})(1))
            min_max_table.(min_max_table.Properties.VariableNames{signal_it})(1) = min_tmp;
        end
        if (max_tmp > min_max_table.(min_max_table.Properties.VariableNames{signal_it})(2))
            min_max_table.(min_max_table.Properties.VariableNames{signal_it})(2) = max_tmp;
        end
    end
end

%% 1. c) Extracting the measurement data 

% Choose what format to save the measurement to the chosen location: 
%   1: *.csv -> for training -> 500 Hz
%   2: *.mat -> timetable -> for simulating in simulink
%   3: *.csv -> for training -> 100 Hz
%   4: *.mat -> timetable -> 100Hz
%   5: *.mat -> normalizing data
extracting_option = 4;

if extracting_option == 1 
    for i = meas_number_list
        location_r = location_read_og + int2str(i) + ".mat";
        file_name = file_name_og + int2str(i) + ".csv";
    
        full_path = fullfile(location_write_og, file_name);
        
        table = extractSimDataSimple(location_r);
        
        writetable(table, full_path, "Delimiter", ','); 
    end

elseif extracting_option == 2
    for i = meas_number_list
        location_r = location_read_og + file_name_og + int2str(i) + ".mat";
        file_name = file_name_og + int2str(i) + ".mat";
    
        full_path = fullfile(location_write_og, file_name);
        
        table = extractSimDataSimple(location_r);
        
        T_meas = table.time_series(end);

        time = seconds(table{:,1});

        time_table = table2timetable(table(:, 2:end), 'RowTimes', time);

        save(full_path, "time_table");
    end
elseif extracting_option == 3
    for i = meas_number_list
        load(location_read_og + file_name_og + int2str(i) + "_norm" + ".mat");
        
        file_name = file_name_og + int2str(i) + "_norm" + ".csv";

        full_path = fullfile(location_write_og, file_name);
        
%         time_table_100 = retime(time_table, 'regular', 'previous', 'TimeStep', seconds(0.01));
% 
        table_100 = timetable2table(time_table_100);

        table_100.Time = seconds(table_100.Time);
        
        writetable(table_100, full_path, "Delimiter", ',');
    end
elseif extracting_option == 4
    for i = meas_number_list
        load(location_read_og + file_name_og + int2str(i) + ".mat");
        
        file_name = file_name_og + int2str(i) + ".mat";

        full_path = fullfile(location_write_og, file_name);
        
        time_table_100 = retime(time_table, 'regular', 'previous', 'TimeStep', seconds(0.01));

        save(full_path, "time_table_100");
    end
elseif extracting_option == 5
    for i = meas_number_list
        load(location_read_og + file_name_og + int2str(i) + ".mat");

        file_name = file_name_og + int2str(i) + "_norm" + ".mat";

        full_path = fullfile(location_write_og, file_name);

        for signal_it = 1:20
            for time_steps_it = 1:length(time_table_100.Time)
                og_value = time_table_100.(time_table_100.Properties.VariableNames{signal_it})(time_steps_it);

                min_value = min_max_table.(time_table_100.Properties.VariableNames{signal_it})(1);
                max_value = min_max_table.(time_table_100.Properties.VariableNames{signal_it})(2);

                norm_value = (og_value - min_value) / (max_value - min_value);
                
                time_table_100.(time_table_100.Properties.VariableNames{signal_it})(time_steps_it) = norm_value;
            end
        end

        save(full_path, "time_table_100");
    end
else
    disp("Please choose a valid option (1, 2, 3 or 4).");
end

%% Creating the time tables with all the reference speeds from the sedan

% have all the files that contain the given number
test_measurements_to_extract = [8 16 31 37 43 49];

main_folder_loc = "C:\work\AI_estimator_dev_toolchain\3_testing\1_model_outputs\it_1";
time_tables_to_save_loc = "C:\work\AI_estimator_dev_toolchain\3_testing\2_time_tables\it_1\time_table_";
file_for_reference = "C:\work\AI_estimator_dev_toolchain\1_data_extraction_for_training\2_extracted_measurements\1_extracted_measurements_mat\ref_1\ref_";

for test_it_time_table = 1:length(test_measurements_to_extract)
    file_paths = findFilesWithNumber(main_folder_loc, test_measurements_to_extract(test_it_time_table));
    
    time_table_to_save = toTimeTable(file_paths, test_measurements_to_extract(test_it_time_table));
    
    % adding the reference speeds
    load(file_for_reference + test_measurements_to_extract(test_it_time_table) + ".mat");
    time_table_to_save = [time_table_to_save, time_table(:, end - 1), time_table(:, end)];
    
    save(time_tables_to_save_loc + test_measurements_to_extract(test_it_time_table) + ".mat", "time_table_to_save");
end

