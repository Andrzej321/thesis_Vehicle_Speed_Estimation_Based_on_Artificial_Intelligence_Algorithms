function [v_bw]= bestWheelSpeedUpdate(u)

omega_w_FL = u(1);
omega_w_FR = u(2);
omega_w_RL = u(3);
omega_w_RR = u(4);

delta_FL = u(5);
delta_FR = u(5);
delta_RL = u(6);
delta_RR = u(6);

w_z = u(7);

acc = u(8);

R = u(9);
r_COG_FL = u(10:12);
r_COG_FR = u(13:15);
r_COG_RL = u(16:18);
r_COG_RR = u(19:21);

prev_sign_acc = u(22);

w = [0; 0; w_z];

% calculating the velocities of the wheels from the wheel rotational speeds
v_w_FL = [omega_w_FL * R; 0; 0];
v_w_FR = [omega_w_FR * R; 0; 0];
v_w_RL = [omega_w_RL * R; 0; 0];
v_w_RR = [omega_w_RR * R; 0; 0];

% transforming the wheel velocities to the coordinate system in the wheels
% parallel to the one in COG
T_FL = [cos(-delta_FL), sin(-delta_FL), 0; -sin(-delta_FL), cos(-delta_FL), 0; 0, 0, 1];
T_FR = [cos(-delta_FR), sin(-delta_FR), 0; -sin(-delta_FR), cos(-delta_FR), 0; 0, 0, 1];
T_RL = [cos(-delta_RL), sin(-delta_RL), 0; -sin(-delta_RL), cos(-delta_RL), 0; 0, 0, 1];
T_RR = [cos(-delta_RR), sin(-delta_RR), 0; -sin(-delta_RR), cos(-delta_RR), 0; 0, 0, 1];

v_w_FL_T = T_FL * v_w_FL;
v_w_FR_T = T_FR * v_w_FR;
v_w_RL_T = T_RL * v_w_RL;
v_w_RR_T = T_RR * v_w_RR;

% transforming the velocities to COG
v_COG_FL = v_w_FL_T + cross(w, r_COG_FL);
v_COG_FR = v_w_FR_T + cross(w, r_COG_FR);
v_COG_RL = v_w_RL_T + cross(w, r_COG_RL);
v_COG_RR = v_w_RR_T + cross(w, r_COG_RR);

% choosing the best wheel

veh_COG_u = [v_COG_FL(1), v_COG_FR(1), v_COG_RL(1), v_COG_RR(1)];

if abs(acc) < eps_bw_acc
    sign_acc = prev_sign_acc;
else
    sign_acc = sign(acc);
end

if sign_acc <= 0
    v_u_bw = max(veh_COG_u);
else
    v_u_bw = min(veh_COG_u);
end

v_bw = [v_u_bw; 0; 0];

end






















