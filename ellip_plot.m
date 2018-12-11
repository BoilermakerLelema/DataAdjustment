function h1 = ellip_plot (xc, yc, a, b, theta, pass)
    x0 = a;
    y0 = 0;
    nseg = 50;
    d_a = 2 * pi / nseg;
    for i = 1 : nseg
        alpha = i * d_a;
        x1 = a * cos(alpha);
        y1 = b * sin(alpha);
        px0 = cos(theta) * x0 - sin(theta) * y0 + xc;
        py0 = sin(theta) * x0 + cos(theta) * y0 + yc;
        px1 = cos(theta) * x1 - sin(theta) * y1 + xc;
        py1 = sin(theta) * x1 + cos(theta) * y1 + yc;
        if ( pass == 1)
            plot ([px0 px1], [py0 py1], 'r-', 'linewidth', 2);
        else
            plot ([px0 px1], [py0 py1], 'g-', 'linewidth', 2);            
        end
        if (i == 1)
            hold on
        end
        x0 = x1;
        y0 = y1;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ellip_plot.m
% CE 597 Adj. Obs.  
% HW 6 20-Nov-2018
% Program Description
% This UDF is used to plot the ellipise
%
% Function Call
% h1 = ellip_plot (x, y, a, b, theta)
%
% Input Arguments
% 1.xc : x-coords of center points
% 2.yc : y-coords of center points
% 3.a : scale
% 4.b : scale
% 5.theta : rotation angle
% 6.pass: 1 = pass global test, 0 = fail global test
%
% Output Arguments
% ellipse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%