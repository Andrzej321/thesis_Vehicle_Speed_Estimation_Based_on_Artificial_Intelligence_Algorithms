function x_plus  = modelBasedSpeedEst(x, u)

% assigning the inputs

% time varying inputs
omega_w_FL = u(1);
omega_w_FR = u(2);
omega_w_RL = u(3);
omega_w_RR = u(4);

delta_FL = u(5);
delta_FR = u(5);
delta_RL = u(6);
delta_RR = u(6);

% parameter inputs
m = u(7);
I_z = u(8);

c_lon_f = u(9);
c_lon_r = u(10);
c_lat_f = u(11);
c_lat_r = u(12);

R = u(13);

% COG -> tire
r_FL = u(14:16);
r_FR = u(17:19);
r_RL = u(20:22);
r_RR = u(23:25);

eps_v_u = u(26);

dt = u(27);

% w_z = u(28);

% assigning the state variables

v_x = x(1, :);
v_y = x(2, :);
w_z = x(3, :);

% to avoid 0 division
% if v_x <= 0.5
%     v_x = 0.5;   
% end

v = [v_x; v_y; 0];
w = [0; 0; w_z];

% transformation matrices
T_FL = [cos(delta_FL), sin(delta_FL), 0; -sin(delta_FL), cos(delta_FL), 0; 0, 0, 1];
T_FL_neg = [cos(-delta_FL), sin(-delta_FL), 0; -sin(-delta_FL), cos(-delta_FL), 0; 0, 0, 1];

T_FR = [cos(delta_FR), sin(delta_FR), 0; -sin(delta_FR), cos(delta_FR), 0; 0, 0, 1];
T_FR_neg = [cos(-delta_FR), sin(-delta_FR), 0; -sin(-delta_FR), cos(-delta_FR), 0; 0, 0, 1];

T_RL = [cos(delta_RL), sin(delta_RL), 0; -sin(delta_RL), cos(delta_RL), 0; 0, 0, 1];
T_RL_neg = [cos(-delta_RL), sin(-delta_RL), 0; -sin(-delta_RL), cos(-delta_RL), 0; 0, 0, 1];

T_RR = [cos(delta_RR), sin(delta_RR), 0; -sin(delta_RR), cos(delta_RR), 0; 0, 0, 1];
T_RR_neg = [cos(-delta_RR), sin(-delta_RR), 0; -sin(-delta_RR), cos(-delta_RR), 0; 0, 0, 1];

% wheel velocity (in the wheel coordinate system) from the vehicle velocity

v_w_FL = T_FL*(v + cross(w, r_FL));
v_w_FR = T_FR*(v + cross(w, r_FR));
v_w_RL = T_RL*(v + cross(w, r_RL));
v_w_RR = T_RR*(v + cross(w, r_RR));

v_w = [v_w_FL, v_w_FR, v_w_RL, v_w_RR];

% Longitdunal slip and sideslip angle calculation

% Denominators (robust to near-zero or sign changes)
den_FL = max(eps_v_u, abs(v_w(1,1)));
den_FR = max(eps_v_u, abs(v_w(1,2)));
den_RL = max(eps_v_u, abs(v_w(1,3)));
den_RR = max(eps_v_u, abs(v_w(1,4)));

% Longitudinal slip
k_FL = -(v_w(1,1) - omega_w_FL*R) / den_FL;
k_FR = -(v_w(1,2) - omega_w_FR*R) / den_FR;
k_RL = -(v_w(1,3) - omega_w_RL*R) / den_RL;
k_RR = -(v_w(1,4) - omega_w_RR*R) / den_RR;

k = [k_FL; k_FR; k_RL; k_RR];

% Lateral side slip angles

alfa_FL = atan2(-v_w(2,1), den_FL);
alfa_FR = atan2(-v_w(2,2), den_FR);
alfa_RL = atan2(-v_w(2,3), den_RL);
alfa_RR = atan2(-v_w(2,4), den_RR);

alfa = [alfa_FL; alfa_FR; alfa_RL; alfa_RR];

% tire forces
F_t_FL_x = k(1) * c_lon_f;
F_t_FR_x = k(2) * c_lon_f;
F_t_RL_x = k(3) * c_lon_r;
F_t_RR_x = k(4) * c_lon_r;

F_t_FL_y = alfa(1) * c_lat_f;
F_t_FR_y = alfa(2) * c_lat_f;
F_t_RL_y = alfa(3) * c_lat_r;
F_t_RR_y = alfa(4) * c_lat_r;

F_t_FL = [F_t_FL_x; F_t_FL_y; 0];
F_t_FR = [F_t_FR_x; F_t_FR_y; 0];
F_t_RL = [F_t_RL_x; F_t_RL_y; 0];
F_t_RR = [F_t_RR_x; F_t_RR_y; 0];

% expressing the COG force and torque
F_t_FL_T = T_FL_neg * F_t_FL;
F_t_FR_T = T_FR_neg * F_t_FR;
F_t_RL_T = T_RL_neg * F_t_RL;
F_t_RR_T = T_RR_neg * F_t_RR;

F_COG = F_t_FL_T + F_t_FR_T + F_t_RL_T + F_t_RR_T;

M_COG = cross(r_FL, F_t_FL_T) + cross(r_FR, F_t_FR_T) + cross(r_RL, F_t_RL_T) + cross(r_RR, F_t_RR_T);

% expressing v_x_dot, v_y_dot and w_z_dot
v_x_dot = F_COG(1)/m + w(3)*v(2);

v_y_dot = F_COG(2)/m - w(3)*v(1);

w_dot = M_COG / I_z;

w_z_dot = w_dot(3);

% forward Euler method for the next step's values

v_x_plus = v_x_dot * dt + v_x;
v_y_plus = v_y_dot * dt + v_y;
w_z_plus = w_z_dot * dt + w_z;

% if v_x_plus < 0
%     v_x_plus = 0;
% end

x_plus = [v_x_plus; v_y_plus; w_z_plus];

end

