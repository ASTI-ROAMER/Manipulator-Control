#include <BasicLinearAlgebra.h>
#include <math.h>
using namespace BLA;

// Dummy angle readings from dynamics4DOF.py
Matrix<100,1>  q1_store = {
     0.00000000e+00, -9.81250049e-04, -9.81249812e-04, -4.13596570e-03,
    -9.64167296e-03, -1.74362326e-02, -2.74663561e-02, -3.96864175e-02,
    -5.40574217e-02, -7.05461090e-02, -8.91241781e-02, -1.09767611e-01,
    -1.32456089e-01, -1.57172483e-01, -1.83902421e-01, -2.12633899e-01,
    -2.43356961e-01, -2.76063414e-01, -3.10746583e-01, -3.47401101e-01,
    -3.86022732e-01, -4.25625052e-01, -4.65219892e-01, -5.04805249e-01,
    -5.44379609e-01, -5.83941862e-01, -6.23491239e-01, -6.63027247e-01,
    -7.02549629e-01, -7.42058310e-01, -7.81553372e-01, -8.21035015e-01,
    -8.60503538e-01, -8.99959311e-01, -9.39402761e-01, -9.78834357e-01,
    -1.01825459e+00, -1.05766399e+00, -1.09706305e+00, -1.13645232e+00,
    -1.17583231e+00, -1.21520353e+00, -1.25456648e+00, -1.29392165e+00,
    -1.33326950e+00, -1.37261049e+00, -1.41194505e+00, -1.45127359e+00,
    -1.49059649e+00, -1.52991414e+00, -1.56922689e+00, -1.60853506e+00,
    -1.64783897e+00, -1.68713891e+00, -1.72643516e+00, -1.76572799e+00,
    -1.80501763e+00, -1.84430430e+00, -1.88358824e+00, -1.92286962e+00,
    -1.96214864e+00, -2.00142548e+00, -2.04070028e+00, -2.07997320e+00,
    -2.11924439e+00, -2.15851396e+00, -2.19778204e+00, -2.23704875e+00,
    -2.27631417e+00, -2.31557842e+00, -2.35484158e+00, -2.39410373e+00,
    -2.43336495e+00, -2.47262531e+00, -2.51188487e+00, -2.55114370e+00,
    -2.59040184e+00, -2.62965937e+00, -2.66891631e+00, -2.70817272e+00,
    -2.74742863e+00, -2.78668409e+00, -2.82471842e+00, -2.86032133e+00,
    -2.89356997e+00, -2.92453104e+00, -2.95326220e+00, -2.97981321e+00,
    -3.00422698e+00, -3.02654042e+00, -3.04678515e+00, -3.06498821e+00,
    -3.08117258e+00, -3.09535771e+00, -3.10755997e+00, -3.11779300e+00,
    -3.12606813e+00, -3.13239465e+00, -3.13678007e+00, -3.13923042e+00
};

Matrix<100,1> q2_store = {
    0.00000000e+00, 3.27250434e-04, 3.27245552e-04, 1.37929348e-03,
    3.21521461e-03, 5.81414497e-03, 9.15832938e-03, 1.32329080e-02,
    1.80257337e-02, 2.35271895e-02, 2.97299903e-02, 3.66289582e-02,
    4.42207664e-02, 5.25036523e-02, 6.14771003e-02, 7.11415015e-02,
    8.14978004e-02, 9.25471408e-02, 1.04290530e-01, 1.16728541e-01,
    1.29861063e-01, 1.43342992e-01, 1.56815328e-01, 1.70276786e-01,
    1.83726185e-01, 1.97162517e-01, 2.10584988e-01, 2.23993033e-01,
    2.37386297e-01, 2.50764610e-01, 2.64127941e-01, 2.77476356e-01,
    2.90809970e-01, 3.04128912e-01, 3.17433291e-01, 3.30723179e-01,
    3.43998599e-01, 3.57259525e-01, 3.70505895e-01, 3.83737622e-01,
    3.96954619e-01, 4.10156825e-01, 4.23344232e-01, 4.36516908e-01,
    4.49675023e-01, 4.62818871e-01, 4.75948883e-01, 4.89065637e-01,
    5.02169865e-01, 5.15262452e-01, 5.28344427e-01, 5.41416952e-01,
    5.54481313e-01, 5.67538890e-01, 5.80591146e-01, 5.93639600e-01,
    6.06685800e-01, 6.19731307e-01, 6.32777667e-01, 6.45826391e-01,
    6.58878938e-01, 6.71936697e-01, 6.85000978e-01, 6.98072996e-01,
    7.11153873e-01, 7.24244628e-01, 7.37346180e-01, 7.50459351e-01,
    7.63584870e-01, 7.76723383e-01, 7.89875461e-01, 8.03041610e-01,
    8.16222284e-01, 8.29417894e-01, 8.42628820e-01, 8.55855421e-01,
    8.69098039e-01, 8.82357007e-01, 8.95632650e-01, 9.08925277e-01,
    9.22235175e-01, 9.35562591e-01, 9.48508611e-01, 9.60682575e-01,
    9.72097517e-01, 9.82765814e-01, 9.92698636e-01, 1.00190570e+00,
    1.01039523e+00, 1.01817406e+00, 1.02524780e+00, 1.03162103e+00,
    1.03729751e+00, 1.04228035e+00, 1.04657216e+00, 1.05017517e+00,
    1.05309131e+00, 1.05532230e+00, 1.05686966e+00, 1.05773476e+00
};

Matrix<100,1> q3_store = {
     0.00000000e+00, -8.87500978e-04, -8.87490032e-04, -3.74066589e-03,
    -8.71978742e-03, -1.57683230e-02, -2.48380767e-02, -3.58885157e-02,
    -4.88861939e-02, -6.38042001e-02, -8.06215916e-02, -9.93227838e-02,
    -1.19896887e-01, -1.42336982e-01, -1.66639342e-01, -1.92802607e-01,
    -2.20826945e-01, -2.50713220e-01, -2.82462213e-01, -3.16073937e-01,
    -3.51547096e-01, -3.87955866e-01, -4.24346829e-01, -4.60717804e-01,
    -4.97067082e-01, -5.33393368e-01, -5.69695712e-01, -6.05973396e-01,
    -6.42225803e-01, -6.78452276e-01, -7.14651988e-01, -7.50823832e-01,
    -7.86966347e-01, -8.23077681e-01, -8.59155599e-01, -8.95197527e-01,
    -9.31200632e-01, -9.67161936e-01, -1.00307844e+00, -1.03894728e+00,
    -1.07476583e+00, -1.11053190e+00, -1.14624378e+00, -1.18190039e+00,
    -1.21750134e+00, -1.25304697e+00, -1.28853834e+00, -1.32397727e+00,
    -1.35936621e+00, -1.39470829e+00, -1.43000710e+00, -1.46526673e+00,
    -1.50049154e+00, -1.53568611e+00, -1.57085515e+00, -1.60600331e+00,
    -1.64113516e+00, -1.67625507e+00, -1.71136714e+00, -1.74647512e+00,
    -1.78158239e+00, -1.81669193e+00, -1.85180628e+00, -1.88692755e+00,
    -1.92205743e+00, -1.95719720e+00, -1.99234775e+00, -2.02750960e+00,
    -2.06268297e+00, -2.09786777e+00, -2.13306364e+00, -2.16827003e+00,
    -2.20348615e+00, -2.23871108e+00, -2.27394373e+00, -2.30918288e+00,
    -2.34442723e+00, -2.37967536e+00, -2.41492576e+00, -2.45017685e+00,
    -2.48542698e+00, -2.52067441e+00, -2.55488411e+00, -2.58701736e+00,
    -2.61710421e+00, -2.64517778e+00, -2.67127181e+00, -2.69541912e+00,
    -2.71765060e+00, -2.73799465e+00, -2.75647686e+00, -2.77311993e+00,
    -2.78794363e+00, -2.80096490e+00, -2.81219800e+00, -2.82165470e+00,
    -2.82934452e+00, -2.83527493e+00, -2.83945163e+00, -2.84187881e+00
};

Matrix<100,1> q4_store = {
     0.00000000e+00, -6.81250766e-04, -6.81242190e-04, -2.87136019e-03,
    -6.69344604e-03, -1.21044049e-02, -1.90679286e-02, -2.75541065e-02,
    -3.75390596e-02, -4.90045527e-02, -6.19375725e-02, -7.63298751e-02,
    -9.21775148e-02, -1.09480363e-01, -1.28241618e-01, -1.48467316e-01,
    -1.70165808e-01, -1.93347208e-01, -2.18022772e-01, -2.44204177e-01,
    -2.71902665e-01, -3.00341262e-01, -3.28622988e-01, -3.56728028e-01,
    -3.84631687e-01, -4.12307417e-01, -4.39729392e-01, -4.66874718e-01,
    -4.93725277e-01, -5.20269224e-01, -5.46502114e-01, -5.72427638e-01,
    -5.98057958e-01, -6.23413656e-01, -6.48523295e-01, -6.73422643e-01,
    -6.98153586e-01, -7.22762799e-01, -7.47300220e-01, -7.71817478e-01,
    -7.96366195e-01, -8.20996390e-01, -8.45754936e-01, -8.70684296e-01,
    -8.95821305e-01, -9.21196304e-01, -9.46832518e-01, -9.72745708e-01,
    -9.98944086e-01, -1.02542849e+00, -1.05219279e+00, -1.07922446e+00,
    -1.10650534e+00, -1.13401247e+00, -1.16171903e+00, -1.18959530e+00,
    -1.21760958e+00, -1.24572911e+00, -1.27392092e+00, -1.30215257e+00,
    -1.33039284e+00, -1.35861221e+00, -1.38678336e+00, -1.41488141e+00,
    -1.44288416e+00, -1.47077220e+00, -1.49852884e+00, -1.52614007e+00,
    -1.55359439e+00, -1.58088253e+00, -1.60799727e+00, -1.63493303e+00,
    -1.66168557e+00, -1.68825164e+00, -1.71462861e+00, -1.74081411e+00,
    -1.76680570e+00, -1.79260056e+00, -1.81819517e+00, -1.84358510e+00,
    -1.86876477e+00, -1.89372731e+00, -1.91814708e+00, -1.94140795e+00,
    -1.96342831e+00, -1.98413645e+00, -2.00347625e+00, -2.02140978e+00,
    -2.03791742e+00, -2.05299604e+00, -2.06665600e+00, -2.07891749e+00,
    -2.08980688e+00, -2.09935328e+00, -2.10758567e+00, -2.11453080e+00,
    -2.12021173e+00, -2.12464710e+00, -2.12785092e+00, -2.12983284e+00
};

//	Define constants
#define pi 3.14159
#define g 9.81

//	Declare functions

//	Dnk - Mass Inertia Terms ---------------------------------------
//		d_nn - inertia moment as seen from the axis of joint "n"
//		d_nk - effect of acceleration of joint "k" on the joint "n"

Matrix<4,4> generateDNK(float d1, float q1, float q2, float q3, float q4, float m1, float m2, float m3, float m4, float a2, float a3, float a4){

	float D11 = 0.08333333333333333 * (d1 * d1) * m1 + 0.08333333333333333 * (a2 * a2) * m2 * (cos(q2) * cos(q2)) + 0.08333333333333333 * (a3 * a3) * m3 * (cos(q2 + q3) * cos(q2 + q3)) + 0.08333333333333333 * (a4 * a4) * m4 * (cos(q2 + q3 + q4) * cos(q2 + q3 + q4)) + m2 * (0.25 * (a2 * a2) * (cos(q1) * cos(q1)) * (cos(q2) * cos(q2)) + 0.25 * (a2 * a2) * (cos(q2) * cos(q2)) * (sin(q1) * sin(q1))) + m3 * (pow((-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))), 2) + pow((0.5 * a3 * cos(q2 + q3) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))), 2)) + m4 * (pow((-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))), 2) + pow((0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))), 2));
	float D12 = m3 * (-((-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) * sin(q1) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3))) - (cos(q1) * (0.5 * a3 * cos(q2 + q3) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3)))) + m4 * (-((-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) * sin(q1) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))) - (cos(q1) * (0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))));
	float D13 = m3 * (-0.5 * a3 * (-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) * sin(q1) * sin(q2 + q3) - 0.5 * a3 * cos(q1) * (0.5 * a3 * cos(q2 + q3) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) * sin(q2 + q3)) + m4 * (-((-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) * sin(q1) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))) - (cos(q1) * (0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))));
	float D14 = m4 * (-0.5 * a4 * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) * sin(q1) * sin(q2 + q3 + q4) - 0.5 * a4 * cos(q1) * (0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1)) * sin(q2 + q3 + q4)));

	float D21 = m3 * (-((-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) * sin(q1) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3))) - (cos(q1) * (0.5 * a3 * cos(q2 + q3) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3)))) + m4 * (-((-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) * sin(q1) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))) - (cos(q1) * (0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))));
	float D22 = 0.08333333333333333 * (a2 * a2) * m2 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a3 * a3) * m3 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a2 * a2) * m2 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a3 * a3) * m3 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m2 * (pow(0.5 * a2 * (cos(q1) * cos(q1) * cos(q2) + cos(q2) * sin(q1) * sin(q1)), 2) + 0.25 * (a2 * a2) * (cos(q1) * cos(q1)) * (sin(q2) * sin(q2)) + 0.25 * (a2 * a2) * (sin(q1) * sin(q1)) * (sin(q2) * sin(q2))) + m3 * (pow(cos(q1) * (-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) + sin(q1) * (-0.5 * a3 * cos(q2 + q3) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)), 2) + (cos(q1) * cos(q1)) * pow(a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3), 2) + (sin(q1) * sin(q1)) * pow(a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3), 2)) + m4 * (pow(cos(q1) * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1)), 2) + (cos(q1) * cos(q1)) * pow(a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4), 2) + (sin(q1) * sin(q1)) * pow(a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4), 2));
	float D23 = 0.08333333333333333 * (a3 * a3) * m3 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a3 * a3) * m3 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m3 * ((cos(q1) * (-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) + sin(q1) * (-0.5 * a3 * cos(q2 + q3) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) * (cos(q1) * (-(a2 * cos(q1) * cos(q2)) -0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) + sin(q1) * (-(a2 * cos(q2) * sin(q1)) -0.5 * a3 * cos(q2 + q3) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) + 0.5 * a3 * (cos(q1) * cos(q1)) * sin(q2 + q3) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3)) + 0.5 * a3 * (sin(q1) * sin(q1)) * sin(q2 + q3) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3))) + m4 * ((cos(q1) * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (cos(q1) * (-(a2 * cos(q1) * cos(q2)) -0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-(a2 * cos(q2) * sin(q1)) -0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) + (cos(q1) * cos(q1)) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) + (sin(q1) * sin(q1)) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)));
	float D24 = 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (1) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (1) + m4 * ((cos(q1) * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (cos(q1) * (-(cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) + 0.5 * a4 * (cos(q1) * cos(q1)) * sin(q2 + q3 + q4) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) + 0.5 * a4 * (sin(q1) * sin(q1)) * sin(q2 + q3 + q4) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)));

	float D31 = m3 * (-0.5 * a3 * (-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) * sin(q1) * sin(q2 + q3) - 0.5 * a3 * cos(q1) * (0.5 * a3 * cos(q2 + q3) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) * sin(q2 + q3)) + m4 * (-((-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) * sin(q1) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))) - (cos(q1) * (0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))));
	float D32 = 0.08333333333333333 * (a3 * a3) * m3 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a3 * a3) * m3 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m3 * ((cos(q1) * (-0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) + sin(q1) * (-0.5 * a3 * cos(q2 + q3) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) * (cos(q1) * (-(a2 * cos(q1) * cos(q2)) - 0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) + sin(q1) * (-(a2 * cos(q2) * sin(q1)) - 0.5 * a3 * cos(q2 + q3) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))) + 0.5 * a3 * (cos(q1) * cos(q1)) * sin(q2 + q3) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3)) + 0.5 * a3 * (sin(q1) * sin(q1)) * sin(q2 + q3) * (a2 * sin(q2) + 0.5 * a3 * sin(q2 + q3))) + m4 * ((cos(q1) * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (cos(q1) * (-(a2 * cos(q1) * cos(q2)) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-(a2 * cos(q2) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) + (cos(q1) * cos(q1)) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) + (sin(q1) * sin(q1)) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)));
	float D33 = 0.08333333333333333 * (a3 * a3) * m3 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a3 * a3) * m3 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m3 * (pow(cos(q1) * (-(a2 * cos(q1) * cos(q2)) - 0.5 * a3 * cos(q1) * cos(q2 + q3) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3)))  + sin(q1) * (-(a2 * cos(q2) * sin(q1)) - 0.5 * a3 * cos(q2 + q3) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)), 2) + 0.25 * (a3 * a3) * (cos(q1) * cos(q1)) * (sin(q2 + q3) * sin(q2 + q3)) + 0.25 * (a3 * a3) * (sin(q1) * sin(q1)) * (sin(q2 + q3) * sin(q2 + q3))) + m4 * (pow(cos(q1) * (-(a2 * cos(q1) * cos(q2)) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)))  + sin(q1) * (-(a2 * cos(q2) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1)), 2) + (cos(q1) * cos(q1)) * pow(a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4), 2) + (sin(q1) * sin(q1)) * pow(a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4), 2));
	float D34 = 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m4 * ((cos(q1) * (-(a2 * cos(q1) * cos(q2)) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4)  + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)))  + sin(q1) * (-(a2 * cos(q2) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1)  + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1)))  * (cos(q1) * (-(cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3)))  - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4)  + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)))  + sin(q1) * (-((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1))  - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1)  + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1)))  + 0.5 * a4 * (cos(q1) * cos(q1)) * sin(q2 + q3 + q4) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4))  + 0.5 * a4 * (sin(q1) * sin(q1)) * sin(q2 + q3 + q4) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)));

	float D41 = m4 * (-0.5 * a4 * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) * sin(q1) * sin(q2 + q3 + q4) - 0.5 * a4 * cos(q1) * (0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) - ((a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * sin(q2 + q3 + q4));;
	float D42 = 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m4 * ((cos(q1) * (-0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (cos(q1) * (-(cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) + 0.5 * a4 * (cos(q1) * cos(q1)) * sin(q2 + q3 + q4) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) + 0.5 * a4 * (sin(q1) * sin(q1)) * sin(q2 + q3 + q4) * (a2 * sin(q2) + a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)));
	float D43 = 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m4 * ((cos(q1) * (-(a2 * cos(q1) * cos(q2)) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-(a2 * cos(q2) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (cos(q1) * (-(cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) + 0.5 * a4 * (cos(q1) * cos(q1)) * sin(q2 + q3 + q4) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)) + 0.5 * a4 * (sin(q1) * sin(q1)) * sin(q2 + q3 + q4) * (a3 * sin(q2 + q3) + 0.5 * a4 * sin(q2 + q3 + q4)));
	float D44 = 0.08333333333333333 * (a4 * a4) * m4 * (cos(q1) * cos(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + 0.08333333333333333 * (a4 * a4) * m4 * (sin(q1) * sin(q1)) * (cos(q1) * cos(q1) + sin(q1) * sin(q1)) + m4 * ((cos(q1) * (-(cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1))) * (cos(q1) * (-(cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3))) - 0.5 * a4 * cos(q1) * cos(q2 + q3 + q4) + cos(q1) * (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4))) + sin(q1) * (-((a2 * cos(q2) + a3 * cos(q2 + q3)) * sin(q1)) - 0.5 * a4 * cos(q2 + q3 + q4) * sin(q1) + (a2 * cos(q2) + a3 * cos(q2 + q3) + a4 * cos(q2 + q3 + q4)) * sin(q1)))) + 0.25 * (a4 * a4) * (cos(q1) * cos(q1)) * (sin(q2 + q3 + q4) * sin(q2 + q3 + q4)) + 0.25 * (a4 * a4) * (sin(q1) * sin(q1)) * (sin(q2 + q3 + q4) * sin(q2 + q3 + q4));

	Matrix<4,4> DNK = {D11, D12, D13, D14,  D21, D22, D23, D24,  D31, D32, D33, D34,  D41, D42, D43, D44};

	return DNK;
}


//	Dnkj - Centrifugal and Coriolis Terms / Christoffel Matrix ---------------------------------------
//		d_nkk · qdot_k^2 - centrifugal effect induced at joint "n" by the velocity of joint "k"
//  	d_nkj · qdot_k·qdot_j - coriolis effect induced at joint "n" by the velocity of joints "k" and "j"
//		d_nnn = 0

Matrix<4,16> generateDNKJ(float d1, float q1, float q2, float q3, float q4, float m1, float m2, float m3, float m4, float a2, float a3, float a4){

	float D111 = 0.0;
	float D112 = 0.1666666666666667 * (((a2 * a2) * m2 + 3 * (a2 * a2) * m3 + 3 * (a2 * a2) * m4) * sin(2 * q2) + (3 * a2 * a3 * m3 + 6 * a2 * a3 * m4) * sin(2 * q2 + q3) + ((a3 * a3) * m3 + 3 * (a3 * a3) * m4) * sin(2 * q2 + 2 * q3) + 3 * a2 * a4 * m4 * sin(2 * q2 + q3 + q4) + 3 * a3 * a4 * m4 * sin(2 * q2 + 2 * q3 + q4) + (a4 * a4) * m4 * sin(2 * q2 + 2 * q3 + 2 * q4));
	float D113 = 0.08333333333333333 * ((3 * a2 * a3 * m3 + 6 * a2 * a3 * m4) * sin(q3) + (3 * a2 * a3 * m3 + 6 * a2 * a3 * m4) * sin(2 * q2 + q3) + (2 * (a3 * a3) * m3 + 6 * (a3 * a3) * m4) * sin(2 * q2 + 2 * q3) + 3 * a2 * a4 * m4 * sin(q3 + q4) + 3 * a2 * a4 * m4 * sin(2 * q2 + q3 + q4) + 6 * a3 * a4 * m4 * sin(2 * q2 + 2 * q3 + q4) + 2 * (a4 * a4) * m4 * sin(2 * q2 + 2 * q3 + 2 * q4));
	float D114 = 0.1666666666666667 * a4 * m4 * (3 * a2 * cos(q2) + 3 * a3 * cos(q2 + q3) + 2 * a4 * cos(q2 + q3 + q4)) * sin(q2 + q3 + q4);

	float D121 = 0.1666666666666667 * ((-((a2 * a2) * m2) - 3 * (a2 * a2) * m3 - 3 * (a2 * a2) * m4) * sin(2 * q2) + (-3 * a2 * a3 * m3 - 6 * a2 * a3 * m4) * sin(2 * q2 + q3) + (-((a3 * a3) * m3) - 3 * (a3 * a3) * m4) * sin(2 * q2 + 2 * q3) - 3 * a2 * a4 * m4 * sin(2 * q2 + q3 + q4) - 3 * a3 * a4 * m4 * sin(2 * q2 + 2 * q3 + q4) - ((a4 * a4) * m4 * sin(2 * q2 + 2 * q3 + 2 * q4)));
	float D122 = 0.0;
	float D123 = 0.0;
	float D124 = 0.0;

	float D131 = 0.08333333333333333 * ( (-3 * a2 * a3 * m3 - 6 * a2 * a3 * m4) * sin(q3) + (-3 * a2 * a3 * m3 - 6 * a2 * a3 * m4) * sin(2 * q2 + q3) + (-2 * (a3 * a3) * m3 - 6 * (a3 * a3) * m4) * sin(2 * q2 + 2 * q3) - 3 * a2 * a4 * m4 * sin(q3 + q4) - 3 * a2 * a4 * m4 * sin(2 * q2 + q3 + q4) - 6 * a3 * a4 * m4 * sin(2 * q2 + 2 * q3 + q4) - 2 * (a4 * a4) * m4 * sin(2 * q2 + 2 * q3 + 2 * q4));
	float D132 = 0.0;
	float D133 = 0.0;
	float D134 = 0.0;

	float D141 = -0.1666666666666667 * a4 * m4 * (3 * a2 * cos(q2) + 3 * a3 * cos(q2 + q3) + 2 * a4 * cos(q2 + q3 + q4)) * sin(q2 + q3 + q4);
	float D142 = 0.0;
	float D143 = 0.0;
	float D144 = 0.0;


	float D211 = 0.1666666666666667 * ( (-(a2 * a2 * m2) - 3 * (a2 * a2) * m3 - 3 * (a2 * a2) * m4) * sin(2 * q2) + (-3 * a2 * a3 * m3 - 6 * a2 * a3 * m4) * sin(2 * q2 + q3) + (-(a3 * a3 * m3) - 3 * (a3 * a3) * m4) * sin(2 * q2 + 2 * q3) - 3 * a2 * a4 * m4 * sin(2 * q2 + q3 + q4) - 3 * a3 * a4 * m4 * sin(2 * q2 + 2 * q3 + q4) - (a4 * a4) * m4 * sin(2 * q2 + 2 * q3 + 2 * q4));
	float D212 = 0.0;
	float D213 = 0.0;
	float D214 = 0.0;

	float D221 = 0.0;
	float D222 = 0.0;
	float D223 = 0.5 * a2 * ( (a3 * m3 + 2 * a3 * m4) * sin(q3) + a4 * m4 * sin(q3 + q4));
	float D224 = 0.5 * a4 * m4 * (a3 * sin(q4) + a2 * sin(q3 + q4));

	float D231 = 0.0;
	float D232 = -0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * sin(q3) + a4 * m4 * sin(q3 + q4));
	float D233 = 0.0;
	float D234 = a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4);

	float D241 = 0.0;
	float D242 = -0.5 * a4 * m4 * (a3 * sin(q4) + a2 * sin(q3 + q4));
	float D243 = -(a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4));
	float D244 = 0.0;


	float D311 = 0.08333333333333333 * ( (-3 * a2 * a3 * m3 - 6 * a2 * a3 * m4) * sin(q3) + (-3 * a2 * a3 * m3 - 6 * a2 * a3 * m4) * sin(2 * q2 + q3) + (-2 * (a3 * a3) * m3 - 6 * (a3 * a3) * m4) * sin(2 * q2 + 2 * q3) - 3 * a2 * a4 * m4 * sin(q3 + q4) - 3 * a2 * a4 * m4 * sin(2 * q2 + q3 + q4) - 6 * a3 * a4 * m4 * sin(2 * q2 + 2 * q3 + q4) - 2 * (a4 * a4) * m4 * sin(2 * q2 + 2 * q3 + 2 * q4));
	float D312 = 0.0;
	float D313 = 0.0;
	float D314 = 0.0;

	float D321 = 0.0;
	float D322 = -0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * sin(q3) + a4 * m4 * sin(q3 + q4));
	float D323 = 0.0;
	float D324 = a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4);

	float D331 = 0.0;
	float D332 = -0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * sin(q3) + a4 * m4 * sin(q3 + q4));;
	float D333 = 0.0;
	float D334 = a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4);;

	float D341 = 0.0;
	float D342 = -0.5 * a4 * m4 * (a3 * sin(q4) + a2 * sin(q3 + q4));
	float D343 = -(a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4));
	float D344 = 0.0;


	float D411 = -0.1666666666666667 * a4 * m4 * (3 * a2 * cos(q2) + 3 * a3 * cos(q2 + q3) + 2 * a4 * cos(q2 + q3 + q4)) * sin(q2 + q3 + q4);;
	float D412 = 0.0;
	float D413 = 0.0;
	float D414 = 0.0;

	float D421 = 0.0;
	float D422 = -0.5 * a4 * m4 * (a3 * sin(q4) + a2 * sin(q3 + q4));;
	float D423 = -(a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4));
	float D424 = 0.0;

	float D431 = 0.0;
	float D432 = -0.5 * a4 * m4 * (a3 * sin(q4) + a2 * sin(q3 + q4));
	float D433 = -(a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4));
	float D434 = 0.0;

	float D441 = 0.0;
	float D442 = -0.5 * a4 * m4 * (a3 * sin(q4) + a2 * sin(q3 + q4));
	float D443 = -(a3 * a4 * m4 * cos(0.5 * q4) * sin(0.5 * q4));
	float D444 = 0.0;


	Matrix<4,16> DNKJ = {D111, D112, D113, D114,  D121, D122, D123, D124,  D131, D132, D133, D134,  D141, D142, D143, D144,
	D211, D212, D213, D214,  D221, D222, D223, D224,  D231, D232, D233, D234,  D241, D242, D243, D244,
	D311, D312, D313, D314,  D321, D322, D323, D324,  D331, D332, D333, D334,  D341, D342, D343, D344,
	D411, D412, D413, D414,  D421, D422, D423, D424,  D431, D432, D433, D434,  D441, D442, D443, D444};

	return DNKJ;
}


//	Gn - Gravity Terms ---------------------------------------

Matrix<4,1> generateGN(float d1, float q1, float q2, float q3, float q4, float m1, float m2, float m3, float m4, float a2, float a3, float a4){

	float G1 = 0.0;
	float G2 = 0.5 * g * ((a2 * m2 + 2 * a2 * m3 + 2 * a2 * m4) * cos(q2) + (a3 * m3 + 2 * a3 * m4) * cos(q2 + q3) + a4 * m4 * cos(q2 + q3 + q4));
	float G3 = 0.5 * g * ((a3 * m3 + 2 * a3 * m4) * cos(q2 + q3) + a4 * m4 * cos(q2 + q3 + q4));
	float G4 = 0.5 * a4 * g * m4 * cos(q2 + q3 + q4);


	Matrix<4,1> GN = {G1, G2, G3, G4};

	return GN;
}



void setup(){
  Serial.begin(2000000);

  // 1. Initialize pins

  // ...Servo - Serial

  // ...Servo PWM

  // ...Distance Sensor

  // ...IMU

  // ...Limit Switches


  // 2. Begin dynamics

  // ... a. Set trajectory duration/length to get to setpoint
  float tfinal = 1.0;

  // ... b. Set number of iterations for splitting the tfinal (granularity)
  float Iterations = tfinal*100;
  float TimeIncrement = pow(10, -2); 

  // ... c. Declare Torque Variables
  float Tau1 = 0.0;
  float Tau2 = 0.0;
  float Tau3 = 0.0;
  float Tau4 = 0.0; 

  // ... d. Declare Error Terms
  float err1 = 0.0;
  float err2 = 0.0;
  float err3 = 0.0;
  float err4 = 0.0;

  float errsum1 = 0.0;
  float errsum2 = 0.0;
  float errsum3 = 0.0;
  float errsum4 = 0.0;

  // ... e. Declare PID Gains - for critically damped condition
  float Kp_term1 = 100.0;
  float Kp_term2 = 100.0;
  float Kp_term3 = 100.0;
  float Kp_term4 = 100.0;

  float Kv_term1 = 20.0;
  float Kv_term2 = 20.0;
  float Kv_term3 = 20.0;
  float Kv_term4 = 20.0;

  BLA::Matrix<4,4> Kv = {Kp_term1, 0, 0, 0,  0, Kp_term2, 0, 0,  0, 0, Kp_term3, 0,  0, 0, 0, Kp_term4};
  BLA::Matrix<4,4> Kp = {Kv_term1, 0, 0, 0,  0, Kv_term2, 0, 0,  0, 0, Kv_term3, 0,  0, 0, 0, Kv_term4};

  // ... f. Set Breakeven Torque ############### add multiplier for r in rxF
  float gm = 9.81;
  float Tau1_max = 0.250*gm;
  float Tau2_max = 0.134*gm;
  float Tau3_max = 0.127*gm;
  float Tau4_max = 0.087*gm;

  // ... g. Set link and joint paramaters
  int N = 4; //Number of joints

  float l1 = 0.180; //Link 1
  float d1 = l1;
  float m1 = 0.250;

  float l2 = 0.300; //Link 2
  float m2 = 0.134;

  float l3 = 0.305; //Link 3
  float m3 = 0.127;

  float m4 = 0.087; //Link 4
  float l4 = 0.071;

  // ... h. Generate Angle, Velocity and Acceleration Trajectories (Piecewise)

  // ....... Set Time Boundaries (Fixed)
  float t1 = 0.2*tfinal; // **0.2 = 2s
  float t2 = 0.8*tfinal; // **0.8 = 8s

  float tc = (t1/tfinal)*(Iterations*TimeIncrement); //2s
  float tm = (t2/tfinal)*(Iterations*TimeIncrement); //8s
  float tf = 1*(Iterations*TimeIncrement); // **1 = 10s

  // ....... Set Angle Setpoints HERE ....HERE....HERE....HERE....

  // radians
  float q1des_main = -3.14;
  float q2des_main = 1.0472;
  float q3des_main = -2.84;
  float q4des_main = -2.18;

  // degrees
  /*
  float q1des_main = 45.0/180.0*pi;
  float q2des_main = 45.0/180.0*pi;
  float q3des_main = 45.0/180.0*pi;
  float q4des_main = 45.0/180.0*pi;
  */

  // ....... Set Max Velocity (Fixed)
  float q1dotdes_max = 2*q1des_main/(2*tf-tc-(tf-tm));
  float q2dotdes_max = 2*q2des_main/(2*tf-tc-(tf-tm));
  float q3dotdes_max = 2*q3des_main/(2*tf-tc-(tf-tm));
  float q4dotdes_max = 2*q4des_main/(2*tf-tc-(tf-tm));

  // ....... Set Max Acceleration (Fixed)
  float q1dotdotdes_max = q1dotdes_max/tc;
  float q2dotdotdes_max = q2dotdes_max/tc;
  float q3dotdotdes_max = q3dotdes_max/tc;
  float q4dotdotdes_max = q4dotdes_max/tc;

  // ....... Set Intersection Points (Fixed)

  float qc1 = 0.5*q1dotdotdes_max*pow(tc,2);
  float qm1 = q1des_main - 0.5*q1dotdotdes_max*pow((tf-tm),2);

  float qc2 = 0.5*q2dotdotdes_max*pow(tc,2);
  float qm2 = q2des_main - 0.5*q2dotdotdes_max*pow((tf-tm),2);

  float qc3 = 0.5*q3dotdotdes_max*pow(tc,2);
  float qm3 = q3des_main - 0.5*q3dotdotdes_max*pow((tf-tm),2);

  float qc4 = 0.5*q4dotdotdes_max*pow(tc,2);
  float qm4 = q4des_main - 0.5*q4dotdotdes_max*pow((tf-tm),2);


  // ... i. Perform Movement with Dynamics
  float q1 = 0.0; //Current Angle
  float q2 = 0.0;
  float q3 = 0.0;
  float q4 = 0.0;
  float q1des = 0.0; //Desired/Target Angle 
  float q2des = 0.0; 
  float q3des = 0.0; 
  float q4des = 0.0;
  float q1dotdes = 0.0;  //Desired/Target Velocity
  float q2dotdes = 0.0; 
  float q3dotdes = 0.0; 
  float q4dotdes = 0.0; 
  float q1dotdotdes = 0.0;  //Desired/Target Acceleration
  float q2dotdotdes = 0.0; 
  float q3dotdotdes = 0.0; 
  float q4dotdotdes = 0.0; 

  // ... j. Compute Matrices Dnk Dnkj and G
  float q1prev = 0.0;
  float q2prev = 0.0;
  float q3prev = 0.0;
  float q4prev = 0.0;

  // ....... /START/ .......
  for ( int o=0; o<Iterations; o++ ) {

    // ....... Timestamp
    unsigned long Tstart = micros();

  	// ....... Get desired setpoints from trajectory generated
  	float tcurr = (o)*TimeIncrement;
  	if (tcurr < tc){ //0s to 2s
      //acceleration
      q1dotdotdes = q1dotdotdes_max;
      q2dotdotdes = q2dotdotdes_max;
      q3dotdotdes = q3dotdotdes_max;
      q4dotdotdes = q4dotdotdes_max;
      
      //velocity
      q1dotdes = q1dotdotdes_max*tcurr;
      q2dotdes = q2dotdotdes_max*tcurr; 
      q3dotdes = q3dotdotdes_max*tcurr;      
      q4dotdes = q4dotdotdes_max*tcurr; 
      
      //angle
      q1des = 0.5*q1dotdotdes_max*pow(tcurr,2);
      q2des = 0.5*q2dotdotdes_max*pow(tcurr,2);
      q3des = 0.5*q3dotdotdes_max*pow(tcurr,2);
      q4des = 0.5*q4dotdotdes_max*pow(tcurr,2);
    }
    else if ((tcurr >= tc) && (tcurr <= tm)){ //between 2s and 8s
      //acceleration
      q1dotdotdes = 0.0;
      q2dotdotdes = 0.0;
      q3dotdotdes = 0.0;
      q4dotdotdes = 0.0;
      
      //velocity
      q1dotdes = q1dotdes_max;
      q2dotdes = q2dotdes_max;
      q3dotdes = q3dotdes_max;
      q4dotdes = q4dotdes_max;
      
      //angle
      q1des = (qm1-qc1)/(tm-tc)*(tcurr) - qc1;
      q2des = (qm2-qc2)/(tm-tc)*(tcurr) - qc2;
      q3des = (qm3-qc3)/(tm-tc)*(tcurr) - qc3;
      q4des = (qm4-qc4)/(tm-tc)*(tcurr) - qc4;

    }
    else{ //8s to 10s
      //acceleration
      q1dotdotdes = - q1dotdotdes_max;
      q2dotdotdes = - q2dotdotdes_max;
      q3dotdotdes = - q3dotdotdes_max;
      q4dotdotdes = - q4dotdotdes_max;
      
      //velocity
      q1dotdes = q1dotdotdes_max*(tf-tcurr); 
      q2dotdes = q2dotdotdes_max*(tf-tcurr); 
      q3dotdes = q3dotdotdes_max*(tf-tcurr);  
      q4dotdes = q4dotdotdes_max*(tf-tcurr); 
      
      //angle
      q1des = q1des_main - 0.5*q1dotdotdes_max*pow(tf-tcurr,2);
      q2des = q2des_main - 0.5*q2dotdotdes_max*pow(tf-tcurr,2);
      q3des = q3des_main - 0.5*q3dotdotdes_max*pow(tf-tcurr,2);
      q4des = q4des_main - 0.5*q4dotdotdes_max*pow(tf-tcurr,2);
    }

    // ....... Read angle position from servos
    //---
    /*
    q1 = 0.0; //y0(1,1) or xdotTele_out[0]
    q2 = 0.0; //y0(1,2) or xdotTele_out[1]
    q3 = 0.0; //y0(1,3) or xdotTele_out[2]
    q4 = 0.0; //y0(1,5) or xdotTele_out[3]
    */

    q1 = q1_store(o,1); //y0(1,1) or xdotTele_out[0]
    q2 = q2_store(o,1); //y0(1,2) or xdotTele_out[1]
    q3 = q3_store(o,1); //y0(1,3) or xdotTele_out[2]
    q4 = q4_store(o,1); //y0(1,5) or xdotTele_out[3]
    //---

    // ....... Compute angular velocity from servos
    float q1dot = (q1-q1prev)/TimeIncrement;//y0(1,5) or xdotTele_out[4]
    float q2dot = (q2-q2prev)/TimeIncrement;//y0(1,6) or xdotTele_out[5]
    float q3dot = (q3-q3prev)/TimeIncrement;//y0(1,7) or xdotTele_out[6]
    float q4dot = (q4-q4prev)/TimeIncrement;//y0(1,8) or xdotTele_out[7]

    // ....... Compute error
    float err1 = q1des - q1;
    float err2 = q2des - q2;
    float err3 = q3des - q3;
    float err4 = q4des - q4;
  
    float err1dot = q1dotdes - q1dot;
    float err2dot = q2dotdes - q2dot;
    float err3dot = q3dotdes - q3dot;
    float err4dot = q4dotdes - q4dot;

    Matrix<4,1> err = {err1 , err2 , err3 , err4};
    Matrix<4,1> errdot = {err1dot , err2dot , err3dot , err4dot};

    // ....... Compute Dnk, Dnkj and Gn Matrices
    
    Matrix<4,4> DNK;
    Matrix<4,16> DNKJ;
    Matrix<4,1> GN;

    DNK = generateDNK(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4);
    DNKJ = generateDNKJ(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4);
    GN = generateGN(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4);

    // ....... Compute desired Dnk, Dnkj and Gn Matrices

    Matrix<4,4> DNK_des;
    Matrix<4,16> DNKJ_des;
    Matrix<4,1> GN_des;

    DNK_des = generateDNK(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4);
    DNKJ_des = generateDNKJ(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4);
    GN_des = generateGN(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4);

    Matrix<4,1> qdotdotDes = {q1dotdotdes , q2dotdotdes , q3dotdotdes , q4dotdotdes};
    Matrix<16,1> qdotDes = {q1dotdes*q1dotdes , q1dotdes*q2dotdes , q1dotdes*q3dotdes , q1dotdes*q4dotdes , q2dotdes*q1dotdes , q2dotdes*q2dotdes , q2dotdes*q3dotdes , q2dotdes*q4dotdes , q3dotdes*q1dotdes , q3dotdes*q2dotdes , q3dotdes*q3dotdes , q3dotdes*q4dotdes , q4dotdes*q1dotdes , q4dotdes*q2dotdes , q4dotdes*q3dotdes , q4dotdes*q4dotdes};

    // ....... Compute output torque

    Matrix<4,1> Tau;
    Tau = (GN_des + DNK_des*qdotdotDes + DNKJ_des*qdotDes) + DNK*(Kv*(errdot)+Kp*(err));

    float Tau1 = Tau(0,0);
    float Tau2 = Tau(1,0);
    float Tau3 = Tau(2,0);
    float Tau4 = Tau(3,0);

    if (Tau1 >= Tau1_max){
      Tau1 = Tau1_max;
    }
    else if (Tau1 < 0){
      Tau1 = 0;
    }
    else {
      Tau1 = Tau1;
    }
  
    if (Tau2 >= Tau2_max){
      Tau2 = Tau2_max;
    }
    else if (Tau2 < 0){
      Tau2 = 0;
    }
    else {
      Tau2 = Tau2;
    } 

    if (Tau3 >= Tau3_max){
      Tau3 = Tau3_max;
    }
    else if (Tau3 < 0){
      Tau3 = 0;
    }
    else {
      Tau3 = Tau3;
    } 

    if (Tau4 >= Tau4_max){
      Tau4 = Tau4_max;
    }
    else if (Tau4 < 0){
      Tau4 = 0;
    }
    else {
      Tau4 = Tau4;
    } 

    // ....... Convert Tau to Angle/PWM output to servo

    


    // ....... Timestamp - get execution time
    unsigned long Tend = micros();
    unsigned long dT = Tend - Tstart;
    float dT_sec = dT/1000000.0;


    // ....... Check dummy data
    //Serial.print("Rad q1");
    Serial.print(q1,6);
    Serial.print(",");
    //Serial.print("Rad q1 ");
    Serial.print(q2,6);
    Serial.print(",");
    //Serial.print("Rad q3");
    Serial.print(q3,6);
    Serial.print(",");
    //Serial.print("Rad q4");
    Serial.print(q3,6);
    Serial.print(",");
    //Serial.print("Iter ");
    Serial.print(o);
    Serial.print(",");
    //Serial.print("dT");
    Serial.print(dT_sec,6);
    Serial.print(",");
    //Serial.print("PWM ");
    //Serial.print(j2_PWM);
    //Serial.print(",");

    //Serial.print("Tau ");
    Serial.println(Tau1);
    //Serial.print(",");
    //Serial.println(q1dot);

  }

}

void loop(){

}
