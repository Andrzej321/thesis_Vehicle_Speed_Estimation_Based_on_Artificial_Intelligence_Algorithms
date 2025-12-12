function x = kinematicVelocity(x, u)

% assigning the inputs
a_x = u(1);
a_y = u(2);
r = u(3);

dt = u(4);

% assigning the state variables
v_u = x(1);
v_v = x(2);

v_u_plus = (a_x + v_v * r) * dt + v_u;
v_v_plus = (a_y - v_u * r) * dt + v_v;

x = [v_u_plus; v_v_plus];

end