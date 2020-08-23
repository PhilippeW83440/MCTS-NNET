A = importdata('profiles_mpcCostU.csv') % Test Case 2

subplot(3,1,1)
plot(A(:,3))
ylim([-5 5])
xlabel('Time t (by steps of 250 ms)')
ylabel('a(t)')
legend('m.s-2')
title('acceleration profile')

subplot(3,1,2)
plot(A(:,2))
ylim([15 30])
xlabel('Time t (by steps of 250 ms)')
ylabel('v(t)')
legend('m.s-1')
title('speed profile')

subplot(3,1,3)
plot(A(:,1))
ylim([0 210])
xlabel('Time t (by steps of 250 ms)')
ylabel('s(t)')
legend('m')
title('path profile')

saveas(gcf, 'profiles.png')
