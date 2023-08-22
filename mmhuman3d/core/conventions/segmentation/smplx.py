"""Raw index information can be found from smpl-wiki website:

https://meshcapade.wiki/SMPL#mesh-templates--samples
"""
SMPLX_SEGMENTATION_DICT = {
    'rightHand':
    [[7331, 7376], [7381, 7386], [7391, 7416], [7419, 7451], [7456], [7459],
     [7463, 7472], [7479, 7494], [7499, 7501], [7504, 7505], [7512, 7514],
     [7520, 7530], [7532, 7535], [7539, 7553], [7556], [7558], [7560, 7564],
     [7571, 7579], [7581], [7585, 7590], [7596, 7610], [7612, 7621],
     [7624, 7629], [7634, 7635], [7637, 7640], [7643], [7947, 7948],
     [7957, 7958], [8047, 8128]],
    'rightUpLeg': [[6225, 6226], [6228, 6229], [6238, 6245], [6261, 6272],
                   [6288, 6306], [6324, 6327], [6335, 6437], [6528, 6539],
                   [6550, 6565], [6575, 6578], [6609, 6618], [6650, 6651],
                   [6662, 6665], [6706, 6707], [6734], [6739, 6746],
                   [6829, 6831], [6833, 6841], [6853, 6857], [6875, 6878],
                   [6888, 6898], [6909, 6910], [8394, 8397], [8400, 8404],
                   [8721], [8725]],
    'leftArm': [[3256, 3259], [3266, 3267], [3311, 3312], [3346, 3349],
                [3401, 3412], [3416, 3425], [3868, 3871], [3898, 3901], [3912],
                [3920, 3921], [3947, 3948], [3951, 3952], [3973, 3976],
                [3987, 3990], [4007, 4031], [4034, 4040], [4042, 4048],
                [4060, 4064], [4067], [4072, 4079], [4135], [4138, 4143],
                [4170, 4174], [4249, 4252], [4261, 4272], [4275, 4278],
                [4281, 4290], [4295, 4296], [4301, 4319], [4322], [4334, 4336],
                [4341, 4358], [4363], [4369, 4373], [4375], [4377, 4378],
                [4383, 4387], [4389], [4398], [4449, 4450], [4460],
                [4464, 4465], [4470, 4471], [4474, 4476], [4478], [4483, 4485],
                [4487, 4489], [4492, 4496], [4500, 4501], [4506, 4507], [4510],
                [4518, 4523], [5397, 5400], [5471, 5479], [5542, 5543],
                [5572, 5573], [5576, 5595], [5597], [5607], [5628]],
    'head': [[0, 11], [16, 218], [223, 371], [376, 461], [464,
                                                          495], [498, 551],
             [554, 557], [560, 562], [565, 648], [651, 735], [738, 1209],
             [1214, 1325], [1327, 1358], [1361, 1385], [1387, 1725],
             [1728, 1758], [1760, 1789], [1791, 1885], [1887, 1897],
             [1899, 1930], [1935, 1939], [1942, 1947], [1950, 2035],
             [2037, 2148], [2152, 2217], [2220, 2483], [2485, 2530],
             [2532, 2869], [2871, 2892], [2894, 2963], [2965, 2975],
             [2977, 3011], [3014, 3183], [8731, 8810], [8815, 8838],
             [8926, 8928], [8931, 8933], [8939], [8941, 8987], [8989, 9019],
             [9028, 9160], [9162, 9164], [9166, 9382]],
    'leftEye': [[9383, 9928]],
    'rightEye': [[9929, 10474]],
    'leftLeg': [[3625, 3626], [3629, 3630], [3635, 3637], [3639], [3642, 3644],
                [3649, 3650], [3675, 3733], [3737, 3769], [3781, 3791],
                [3809, 3817], [3999, 4001], [4003, 4006], [4098, 4108],
                [4154, 4164], [5728, 5764], [5873, 5889], [8892, 8896],
                [8935, 8937], [9020]],
    'leftToeBase': [[5765, 5872], [5890], [5893], [5895], [5897], [5899],
                    [5901], [5903, 5904], [5906], [5908], [5911, 5912], [5914],
                    [5916]],
    'leftFoot': [[5881, 5919], [5922, 5930], [5933], [8728, 8730],
                 [8839, 8925], [8929, 8930], [8934, 8935]],
    'spine1':
    [[3228, 3231], [3240, 3251], [3272, 3273], [3276, 3277], [3282, 3283],
     [3288, 3291], [3298, 3301], [3314, 3322], [3352], [3355, 3357], [3369],
     [3383, 3384], [3393, 3394], [3399, 3400], [3426, 3427], [3521, 3524],
     [3555, 3559], [3570, 3573], [3824, 3830], [3833], [3836, 3838], [3844],
     [3855, 3856], [3873], [3892, 3893], [3896, 3897], [3908, 3910],
     [3981, 3982], [3985], [4052, 4054], [4056, 4058], [4069, 4070],
     [4392, 4394], [5417, 5429], [5448, 5449], [5459], [5483], [5485, 5486],
     [5489], [5531, 5532], [5534], [5632], [5634, 5635], [5638, 5639], [5642],
     [5644, 5648], [5944], [5950], [5991, 5994], [6003, 6014], [6035, 6036],
     [6039, 6040], [6045, 6046], [6051, 6054], [6061, 6064], [6077, 6085],
     [6115, 6118], [6130], [6144, 6145], [6154, 6155], [6160, 6161],
     [6187, 6188], [6282, 6285], [6316, 6320], [6331, 6334], [6581, 6588],
     [6591, 6593], [6599], [6624], [6640, 6641], [6644, 6645], [6656, 6658],
     [6729, 6730], [6733], [6798, 6800], [6802, 6804], [6813, 6814],
     [7128, 7130], [8151, 8163], [8182, 8183], [8193], [8217, 8218], [8326],
     [8328, 8329], [8332, 8333], [8336], [8338, 8342], [8726], [9026]],
    'spine2': [[3210, 3211], [3214, 3227], [3232, 3239], [3252, 3255],
               [3268, 3271], [3274, 3275], [3278, 3281], [3296, 3297],
               [3302, 3305], [3310, 3313], [3323, 3334], [3342, 3343],
               [3345, 3347], [3358, 3365], [3367, 3368], [3370, 3382],
               [3385, 3392], [3395, 3396], [3435, 3438], [3443, 3446],
               [3449, 3453], [3525, 3526], [3560, 3561], [3831, 3832],
               [3834, 3835], [3846, 3850], [3853, 3854], [3857], [3872],
               [3874], [3894, 3895], [3911, 3913], [3922, 3946], [3979, 3980],
               [3983, 3984], [4032], [4049, 4051], [4055], [4059], [4068],
               [4071], [4136, 4137], [4168, 4169], [4174, 4175], [4279, 4280],
               [4391], [4395, 4399], [4426, 4429], [4434, 4438], [4452, 4457],
               [4486], [4497, 4498], [5349, 5350], [5395, 5396], [5430, 5447],
               [5450], [5453, 5454], [5457, 5458], [5460, 5462], [5480, 5482],
               [5484], [5487], [5499, 5501], [5519], [5521,
                                                      5526], [5528, 5530],
               [5533], [5536], [5547, 5556], [5558, 5571], [5598, 5599],
               [5611, 5612], [5618, 5619], [5621], [5633], [5636, 5637],
               [5640, 5641], [5643], [5650, 5657], [5920, 5921], [5932],
               [5935, 5938], [5945], [5947], [5973, 5974], [5977, 5990],
               [5995, 6002], [6015, 6018], [6031, 6034], [6037, 6038],
               [6041, 6044], [6059, 6060], [6065, 6068], [6073, 6076],
               [6086, 6097], [6105, 6106], [6108, 6110], [6119, 6126],
               [6128, 6129], [6131, 6143], [6146, 6153], [6156, 6157],
               [6196, 6199], [6204, 6207], [6210, 6214], [6286, 6287],
               [6321, 6322], [6589, 6590], [6601, 6608], [6623], [6625],
               [6642, 6643], [6659, 6661], [6670, 6694], [6727, 6728],
               [6731, 6732], [6779], [6795, 6797], [6801], [6805], [6812],
               [6815], [6880, 6881], [6912, 6913], [6918, 6919], [7127],
               [7131, 7135], [7162, 7165], [7170, 7174], [7188, 7193], [7222],
               [7233, 7234], [8129, 8130], [8164, 8181], [8184], [8187, 8188],
               [8191, 8192], [8194, 8196], [8214, 8216], [8241, 8247], [8249],
               [8260, 8283], [8307, 8308], [8316, 8317], [8327], [8330, 8331],
               [8334, 8335], [8337], [8344, 8351], [8727], [9027]],
    'leftShoulder': [[3219], [3233, 3234], [3236, 3237], [3264, 3267], [3303],
                     [3336, 3341], [3343, 3346], [3362, 3363], [3366, 3367],
                     [3413, 3415], [3875, 3878], [3880, 3883], [3929, 3930],
                     [3935], [3953, 3955], [4032, 4035], [4143], [4167],
                     [4174], [4426, 4428], [4430, 4433], [4436], [4438, 4451],
                     [4455], [4458, 4477], [4479, 4482], [4490, 4491],
                     [4498, 4499], [4502, 4505], [4508, 4509], [4511, 4517],
                     [5455, 5457], [5462, 5470], [5479], [5535, 5546],
                     [5563, 5564], [5566], [5602], [5605, 5610], [5624, 5627]],
    'rightShoulder': [[5982], [5996, 5997], [5999, 6000], [6027, 6030], [6066],
                      [6099, 6104], [6106, 6109], [6123, 6124], [6127, 6128],
                      [6174, 6176], [6626, 6633], [6677, 6678], [6683],
                      [6701, 6703], [6779, 6782], [6887], [6911], [6918],
                      [7162, 7164], [7166, 7169], [7172], [7174, 7187], [7191],
                      [7194, 7213], [7215, 7218], [7226, 7227], [7234, 7235],
                      [7238, 7241], [7244, 7245], [7247, 7253], [8189, 8191],
                      [8196, 8204], [8213], [8248, 8259], [8275, 8276], [8278],
                      [8309, 8315], [8318, 8321]],
    'rightFoot': [[8575, 8717]],
    'rightArm':
    [[6019, 6022], [6029, 6030], [6074, 6075], [6109, 6112], [6162, 6173],
     [6177, 6186], [6619, 6622], [6646, 6649], [6660], [6668, 6669],
     [6695, 6696], [6699, 6700], [6721, 6724], [6735, 6738], [6754, 6778],
     [6781, 6794], [6806, 6811], [6816, 6823], [6879], [6882, 6887],
     [6914, 6918], [6993, 6996], [7005, 7016], [7019, 7032], [7035, 7036],
     [7039, 7058], [7070, 7072], [7077, 7094], [7099], [7105, 7109], [7111],
     [7113, 7114], [7119, 7123], [7125], [7134], [7185, 7186], [7196],
     [7200, 7201], [7206, 7207], [7210, 7212], [7214], [7219, 7221],
     [7223, 7225], [7228, 7232], [7236, 7237], [7242, 7243], [7246],
     [7254, 7259], [8131, 8134], [8205, 8213], [8255, 8256], [8284, 8306],
     [8312], [8322]],
    'leftHandIndex1': [[4641, 4644], [4651, 4654], [4669], [4681, 4682],
                       [4737, 4745], [4759, 4760], [4766, 4768], [4770, 4783],
                       [4791, 4793], [4795], [4800, 4802], [4805],
                       [4818, 4819], [4829, 4834], [4846, 4847], [4859, 4861],
                       [4872], [4874, 4877], [4883, 4884], [4886, 4888],
                       [4890, 4891], [4894, 4897], [4905, 5210], [5213, 5220],
                       [5223, 5310]],
    'rightLeg': [[6386, 6387], [6390, 6391], [6396, 6398], [6400],
                 [6403, 6405], [6410, 6411], [6436, 6527], [6539, 6549],
                 [6566, 6574], [6747, 6753], [6842, 6852], [6898, 6908],
                 [8422, 8458], [8567, 8583], [8680, 8684], [8717, 8720]],
    'rightHandIndex1': [[7377, 7380], [7387, 7390], [7405], [7417, 7418],
                        [7473, 7481], [7495, 7496], [7502, 7504], [7506, 7519],
                        [7527, 7529], [7531], [7536, 7538], [7541],
                        [7554, 7555], [7565, 7570], [7582, 7583], [7595, 7597],
                        [7608], [7610, 7613], [7619, 7620], [7622, 7624],
                        [7626, 7627], [7630, 7633], [7641, 7946], [7949, 7956],
                        [7959, 8046]],
    'leftForeArm': [[4176, 4248], [4251, 4260], [4273, 4274], [4277, 4278],
                    [4283, 4284], [4287, 4290], [4293, 4296], [4299, 4302],
                    [4323, 4333], [4337, 4340], [4359, 4368], [4371], [4374],
                    [4376], [4379, 4382], [4388], [4390], [4518], [4523, 4594],
                    [4632], [4673, 4674], [4686], [4703], [4712, 4726],
                    [4761, 4762], [4820, 4823], [4842], [4844], [4848, 4849],
                    [4855, 4858], [4893], [4900], [5451, 5452]],
    'rightForeArm': [[6920, 6992], [6995, 7004], [7017, 7018], [7021, 7022],
                     [7025, 7026], [7029, 7040], [7059, 7069], [7073, 7076],
                     [7095, 7104], [7107],
                     [7110], [7112], [7115, 7118], [7124], [7126], [7254],
                     [7259, 7330], [7368], [7409, 7410], [7422], [7439],
                     [7448, 7462], [7497, 7498], [7556, 7559], [7578], [7580],
                     [7584, 7585], [7591, 7594], [7629], [7636], [8185, 8186]],
    'neck': [[12, 15], [219, 222], [372, 375], [462, 463], [496, 497],
             [552, 553], [558, 559], [563, 564], [649, 650], [736, 737],
             [1210, 1213], [1326], [1359, 1360], [1386], [1726, 1727], [1759],
             [1790], [1886], [1898], [1931, 1934], [1940, 1941], [1948, 1949],
             [2036], [2149, 2151], [2218, 2219], [2484], [2531], [2870],
             [2893], [2964], [2976], [3012, 3013], [3184, 3213], [3353, 3354],
             [3435, 3436], [3445, 3446], [3450], [3452, 3453], [3456, 3459],
             [3857], [3918, 3919], [3944, 3945], [3949, 3950], [3956, 3957],
             [3964], [5518, 5519], [5527], [5616, 5617], [5649], [5920],
             [5951, 5976], [6196, 6197], [6206, 6207], [6211], [6213, 6214],
             [6217, 6220], [6608], [6666, 6667], [6692, 6693], [6697, 6698],
             [6704, 6705], [6712], [8343], [8938], [8940], [8988]],
    'rightToeBase': [[8459, 8566], [8584], [8587], [8589], [8591], [8593],
                     [8595], [8597, 8598], [8600], [8602], [8605, 8606],
                     [8608], [8610]],
    'spine': [[3244, 3245], [3260, 3263], [3284, 3287], [3292, 3295],
              [3350, 3351], [3397, 3400], [3428, 3431], [3519, 3520],
              [3546, 3547], [3549, 3556], [3822, 3823], [3844, 3845],
              [3851, 3852], [3886, 3888], [3891], [3904, 3907], [3960, 3963],
              [3965, 3968], [3970], [3977, 3978], [4114, 4129], [4400],
              [5401, 5423], [5425, 5426], [5429], [5488, 5489], [5495, 5496],
              [5623], [5629, 5631], [5699], [5939, 5941], [5943], [5948],
              [5950], [6007, 6008], [6023, 6026], [6047, 6050], [6055, 6058],
              [6113, 6114], [6158, 6161], [6189, 6192], [6280, 6281],
              [6307, 6308], [6310, 6317], [6579, 6580], [6599, 6600],
              [6636, 6639], [6652, 6655], [6708, 6711], [6713, 6716], [6718],
              [6725, 6726], [6858, 6873], [7136], [8135, 8157], [8159, 8160],
              [8163], [8323, 8325], [8393], [8722, 8724], [8726], [9022, 9024],
              [9026]],
    'leftUpLeg': [[3464, 3465], [3467, 3468], [3477, 3484], [3500, 3511],
                  [3527, 3545], [3563, 3566], [3574, 3676], [3770, 3781],
                  [3792, 3803], [3805, 3808], [3818, 3821], [3858, 3867],
                  [3902, 3903], [3914, 3917], [3958, 3959], [3986],
                  [3991, 3998], [4085, 4087], [4089, 4097], [4109, 4113],
                  [4131, 4134], [4144, 4154], [4165, 4166], [5700, 5703],
                  [5706, 5710], [9021], [9025]],
    'eyeballs': [[9383, 9516], [9518, 9529], [9531, 9542], [9544, 9555],
                 [9557, 9568], [9570, 9581], [9583, 9594], [9596, 9607],
                 [9609, 9620], [9622, 9633], [9635, 9646], [9648, 9659],
                 [9661, 9672], [9674, 9685], [9687, 9698], [9700, 9711],
                 [9713, 9724], [9726, 9737], [9739, 9750], [9752, 9763],
                 [9765, 9776], [9778, 9789], [9791, 9803], [9805, 9816],
                 [9818, 9829], [9831, 9842], [9844, 9855], [9857, 9868],
                 [9870, 9881], [9883, 9894], [9896, 9907], [9909, 9920],
                 [9922, 10062], [10064, 10075], [10077, 10088], [10090, 10101],
                 [10103, 10114], [10116, 10127], [10129,
                                                  10140], [10142, 10153],
                 [10155, 10166], [10168, 10179], [10181,
                                                  10192], [10194, 10205],
                 [10207, 10218], [10220, 10231], [10233,
                                                  10244], [10246, 10257],
                 [10259, 10270], [10272, 10283], [10285,
                                                  10296], [10298, 10309],
                 [10311, 10322], [10324, 10335], [10337,
                                                  10349], [10351, 10362],
                 [10364, 10375], [10377, 10388], [10390,
                                                  10401], [10403, 10414],
                 [10416, 10427], [10429, 10440], [10442, 10453],
                 [10455, 10466], [10468, 10474]],
    'leftHand': [[4595, 4640],
                 [4645, 4650], [4655, 4680], [4683, 4715], [4720], [4723],
                 [4727, 4736], [4743, 4758], [4763, 4765], [4768, 4769],
                 [4776, 4778], [4784, 4794], [4796, 4799], [4803,
                                                            4817], [4820],
                 [4822], [4824, 4828], [4835, 4843], [4845], [4849, 4854],
                 [4860, 4874], [4876, 4885], [4888, 4893], [4898, 4899],
                 [4901, 4904], [4907], [5211, 5212], [5221,
                                                      5222], [5311, 5348],
                 [5351, 5394]],
    'hips': [[3262, 3263], [3284, 3285], [3292, 3293], [3306, 3309], [3335],
             [3350], [3428, 3429], [3432, 3434], [3439, 3442], [3447, 3448],
             [3454, 3455], [3460, 3476], [3485, 3500], [3510, 3520],
             [3542, 3543], [3546, 3550], [3562], [3567, 3569], [3734, 3736],
             [3798, 3799], [3804], [3839, 3843], [3879], [3884, 3885],
             [3889, 3890], [3902, 3903], [3916, 3917], [3958], [3969, 3972],
             [3986], [3993, 3994], [4002], [4041], [4065, 4066], [4080, 4084],
             [4088], [4130], [4144, 4145], [4147], [4165, 4166], [4291, 4292],
             [4297, 4298], [4320, 4321], [4401, 4425], [5490, 5494],
             [5497, 5498], [5502, 5517], [5520], [5557], [5574, 5575], [5596],
             [5600, 5601], [5603, 5604], [5613, 5615], [5620], [5622],
             [5630, 5631], [5658, 5699], [5703, 5705], [5711, 5727], [5931],
             [5934], [5939], [5941, 5942], [5946], [5949], [6025, 6026],
             [6047, 6048], [6055, 6056], [6069, 6072], [6098], [6113],
             [6189, 6190], [6193, 6195], [6200, 6203], [6208, 6209],
             [6215, 6216], [6221, 6237], [6246, 6261], [6271, 6281],
             [6303, 6304], [6307, 6311], [6323], [6328, 6330], [6556, 6557],
             [6594, 6598], [6634, 6635], [6650, 6651], [6664, 6665], [6706],
             [6717, 6720], [6734], [6741, 6742], [6824, 6828], [6832], [6874],
             [6888, 6889], [6891], [6909, 6910], [7137, 7161], [8219, 8240],
             [8324, 8325], [8352, 8393], [8397, 8399], [8405, 8421]]
}

SMPLX_SUPER_SET = {
    'FOOT': ['leftFoot', 'leftToeBase', 'rightFoot', 'rightToeBase'],
    'HAND': ['leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1'],
    'LEG': ['rightUpLeg', 'leftUpLeg', 'leftLeg', 'rightLeg'],
    'ARM': ['leftForeArm', 'rightForeArm', 'leftArm', 'rightArm'],
    'HEAD': ['neck', 'head', 'leftEye', 'rightEye', 'eyeballs'],
    'UPBODY': ['spine1', 'spine2', 'leftShoulder', 'rightShoulder'],
    'LOWBODY': ['spine', 'hips'],
}
