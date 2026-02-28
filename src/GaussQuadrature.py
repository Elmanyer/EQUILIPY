# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Author: Pau Manyer Fuertes
# Email: pau.manyer@bsc.es
# Date: July 2025
# Institution: Barcelona Supercomputing Center (BSC)
# Department: Computer Applications in Science and Engineering (CASE)
# Research Group: Nuclear Fusion  


# This script contains the numerical integration Gauss quadratures for 
# different types of elements and different orders of quadrature. 

import numpy as np

def GaussQuadrature(element,order):
    """ Obtain Gauss quadrature for reference element and selected quadrature order. 
        Input: - element: type of element -> 0=line, 1=tri, 2=quad 
               - order: quadrature order 
        Output: - z: Gauss nodal coordinates matrix 
                - w: Gauss integration weights """
    match element:
            case 0:  # LINE (1D ELEMENT)
                match order:
                    case 1:
                        Ng = 1
                        zg = 0
                        wg = 2
                    case 2:
                        Ng = 2
                        sq = 1/np.sqrt(3)
                        zg = np.array([-sq, sq])
                        zg = np.reshape(zg, (Ng,1))
                        wg = np.ones([Ng])
                    case 3:
                        Ng = 3
                        sq = np.sqrt(3/5)
                        zg = np.array([-sq, 0, sq])
                        zg = np.reshape(zg, (Ng,1))
                        wg = np.array([5/9, 8/9, 5/9])
                    case 4:
                        Ng = 4
                        sq1 = np.sqrt(3/7-(2/7)*np.sqrt(6/5))
                        sq2 = np.sqrt(3/7+(2/7)*np.sqrt(6/5))
                        w1 = (18+np.sqrt(30))/36
                        w2 = (18-np.sqrt(30))/36
                        zg = np.array([-sq2,-sq1,sq1,sq2])
                        zg = np.reshape(zg, (Ng,1))
                        wg = np.array([w2,w1,w1,w2])
                    case 5:
                        Ng = 5
                        sq1 = (1/3)*np.sqrt(5-2*np.sqrt(10/7))
                        sq2 = (1/3)*np.sqrt(5+2*np.sqrt(10/7))
                        w1 = (322+13*np.sqrt(70))/900
                        w2 = (322-13*np.sqrt(70))/900
                        zg = np.array([-sq2,-sq1,0,sq1,sq2])
                        zg = np.reshape(zg, (Ng,1))
                        wg = np.array([w2,w1,128/225,w1,w2])
                        
            case 1:  # TRIANGLE
                match order:
                    case 1:
                        Ng = 1
                        zg = np.array([1/3, 1/3])
                        wg = 1/2
                    case 2:   
                        Ng = 3
                        zg = np.zeros([Ng,2])
                        #zg[0,:] = [0.5, 0.5]
                        #zg[1,:] = [0, 0.5]
                        #zg[2,:] = [0.5, 0]
                        zg[0,:] = [1/6, 2/3]
                        zg[1,:] = [1/6, 1/6]
                        zg[2,:] = [2/3, 1/6]
                        wg = np.ones(Ng)*(1/6)
                    case 3:  
                        Ng = 4  
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [0.2, 0.2]
                        zg[1,:] = [0.6, 0.2]
                        zg[2,:] = [0.2, 0.6]
                        zg[3,:] = [1/3, 1/3]
                        wg = np.array([25/96, 25/96, 25/96, -27/96])
                    case 4: 
                        Ng = 6 
                        a = 0.108103018168070227360
                        b = 0.445948490915964886320
                        c = 0.816847572980458513080
                        d = 0.091576213509770743460
                        w1 = 0.22338158967801146570
                        w2 = 0.10995174365532186764
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, b]
                        zg[1,:] = [b, b]
                        zg[2,:] = [b, a]
                        zg[3,:] = [c, d]
                        zg[4,:] = [d, d]
                        zg[5,:] = [d, c]
                        wg = np.array([w1, w1, w1, w2, w2, w2])*0.5
                    case 5:
                        Ng = 7
                        a = 0.333333333333333333333
                        b = 0.470142064105115089770
                        c = 0.059715871789769820459
                        d = 0.101286507323456338800
                        e = 0.797426985353087322400
                        w1 = 0.225
                        w2 = 0.12593918054482715260
                        w3 = 0.13239415278850618074
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, a]
                        zg[1,:] = [b, c]
                        zg[2,:] = [b, b]
                        zg[3,:] = [c, b]
                        zg[4,:] = [d, e]
                        zg[5,:] = [d, d]
                        zg[6,:] = [e, d]
                        wg = np.array([w1, w2, w2, w2, w3, w3, w3])*0.5
                        
                    case 6:
                        Ng = 12
                        zg = np.zeros([Ng,2])
                        a = 0.063089014491502
                        b = 0.873821971016996
                        c = 0.053145049844817
                        d = 0.636502499121399
                        e = 0.310352451033784
                        f = 0.249286745170910
                        g = 0.501426509658179
                        w1= 0.050844906370207
                        w2= 0.082851075618374
                        w3= 0.116786275726379
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, b]
                        zg[1,:] = [a, a]         
                        zg[2,:] = [b, a]
                        zg[3,:] = [c, d]
                        zg[4,:] = [e, c]
                        zg[5,:] = [d, e]        
                        zg[6,:] = [e, d]
                        zg[7,:] = [c, e]        
                        zg[8,:] = [d, c]
                        zg[9,:] = [f, g]        
                        zg[10,:] = [f, f]
                        zg[11,:] = [g, f]        
                        wg = np.array([w1, w1, w1, w2, w2, w2, w2, w2, w2, w3, w3, w3])*0.5
                        
                    case 7:
                        Ng = 13
                        a = 0.333333333333333
                        b = 0.479308067841920
                        c = 0.869739794195568
                        d = 0.638444188569810
                        e = 0.260345966079040
                        f = 0.065130102902216
                        g = 0.312865496004874
                        h = 0.048690315425316
                        w1=-0.149570044467670
                        w2= 0.175615257433204
                        w3= 0.053347235608839
                        w4= 0.077113760890257
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, a]
                        zg[1,:] = [e, e]         
                        zg[2,:] = [b, e]
                        zg[3,:] = [e, b]
                        zg[4,:] = [f, f]
                        zg[5,:] = [c, f]        
                        zg[6,:] = [f, c]
                        zg[7,:] = [d, g]        
                        zg[8,:] = [d, h]
                        zg[9,:] = [g, d]        
                        zg[10,:] = [g, h]
                        zg[11,:] = [h, d]        
                        zg[12,:] = [h, g]
                        wg = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w4, w4, w4])*0.5
                        
                    case 8:
                        Ng = 16
                        a = 0.333333333333333
                        b = 0.459292588292723
                        c = 0.081414823414554
                        d = 0.170569307751760
                        e = 0.658861384496480
                        f = 0.008394777409958
                        g = 0.728492392955404
                        h = 0.263112829634638
                        i = 0.050547228317031
                        j = 0.898905543365938
                        w1= 0.144315607677787
                        w2= 0.095091634267285
                        w3= 0.103217370534718
                        w4= 0.027230314174435
                        w5= 0.032458497623198
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, a]
                        zg[1,:] = [b, c]         
                        zg[2,:] = [b, b]
                        zg[3,:] = [c, b]
                        zg[4,:] = [d, e]
                        zg[5,:] = [d, d]        
                        zg[6,:] = [e, d]
                        zg[7,:] = [f, g]        
                        zg[8,:] = [h, f]
                        zg[9,:] = [g, h]        
                        zg[10,:] = [h, g]
                        zg[11,:] = [f, h]        
                        zg[12,:] = [g, f]
                        zg[13,:] = [i, j]
                        zg[14,:] = [i, i]
                        zg[15,:] = [j, i]
                        wg = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w4, w4, w4, w5, w5, w5])*0.5
                    
                    case 9:
                        Ng = 19
                        a = 0.333333333333333
                        b = 0.489682519198738
                        c = 0.020634961602525
                        d = 0.437089591492937
                        e = 0.125820817014127
                        f = 0.188203535619033
                        g = 0.623592928761935
                        h = 0.036838412054736
                        i = 0.741198598784498
                        j = 0.221962989160766
                        k = 0.044729513394453
                        l = 0.910540973211095
                        w1= 0.097135796282799
                        w2= 0.031334700227139
                        w3= 0.077827541004774
                        w4= 0.079647738927210
                        w5= 0.043283539377289
                        w6= 0.025577675658698
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, a]
                        zg[1,:] = [b, c]         
                        zg[2,:] = [b, b]
                        zg[3,:] = [c, b]
                        zg[4,:] = [d, e]
                        zg[5,:] = [d, d]        
                        zg[6,:] = [e, d]
                        zg[7,:] = [f, g]        
                        zg[8,:] = [f, f]
                        zg[9,:] = [g, f]        
                        zg[10,:] = [h, i]
                        zg[11,:] = [j, h]        
                        zg[12,:] = [i, j]
                        zg[13,:] = [j, i]
                        zg[14,:] = [h, j]
                        zg[15,:] = [i, h]
                        zg[16,:] = [k, l]
                        zg[17,:] = [k, k]
                        zg[18,:] = [l, k]
                        wg = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w5, w5, w5, w5, w5, w5, w6, w6, w6])*0.5
                        
                    case 10:
                        Ng = 25
                        a = 0.333333333333333
                        b = 0.485577633383657
                        c = 0.028844733232685
                        d = 0.141707219414880
                        e = 0.550352941820999
                        f = 0.307939838764121
                        g = 0.025003534762686
                        h = 0.728323904597411
                        i = 0.246672560639903
                        j = 0.009540815400299
                        k = 0.923655933587500
                        l = 0.066803251012200
                        m = 0.109481575485037
                        n = 0.781036849029926
                        w1= 0.090817990382754
                        w2= 0.036725957756467
                        w3= 0.072757916845420
                        w4= 0.028327242531057
                        w5= 0.009421666963733
                        w6= 0.045321059435528
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a, a]
                        zg[1,:] = [b, c]         
                        zg[2,:] = [b, b]
                        zg[3,:] = [c, b]
                        zg[4,:] = [d, e]
                        zg[5,:] = [f, d]        
                        zg[6,:] = [e, f]
                        zg[7,:] = [f, e]        
                        zg[8,:] = [d, f]
                        zg[9,:] = [e, d]        
                        zg[10,:] = [g, h]
                        zg[11,:] = [i, g]        
                        zg[12,:] = [h, i]
                        zg[13,:] = [i, h]
                        zg[14,:] = [g, i]
                        zg[15,:] = [h, g]
                        zg[16,:] = [j, k]
                        zg[17,:] = [l, j]
                        zg[18,:] = [k, l]
                        zg[19,:] = [l, k]
                        zg[20,:] = [j, l]
                        zg[21,:] = [k, j]
                        zg[22,:] = [m, n]
                        zg[23,:] = [m, m]
                        zg[24,:] = [n, m]
                        wg = np.array([w1, w2, w2, w2, w3, w3, w3, w3, w3, w3, w4, w4, w4, w4, w4, w4, w5, w5, w5, w5, w5, w5, w6, w6, w6])*0.5
                               
            case 2:  # QUADRILATERAL
                match order:
                    case 1:
                        Ng = 1
                        zg = np.zeros([2])
                        wg = 4
                    case 2:
                        Ng = 4
                        zg = np.zeros([Ng,2])
                        a = 1/np.sqrt(3)
                        zg[0,:] = [-a, -a]
                        zg[1,:] = [a, -a]
                        zg[2,:] = [a, a]
                        zg[2,:] = [-a, a]
                        wg = np.ones(Ng)
                    case 3:  
                        Ng = 9
                        a = 0.774596669241483
                        w1 = 0.308641975308641
                        w2 = 0.493827160493826
                        w3 = 0.790123456790123
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a,a]
                        zg[1,:] = [a,0]
                        zg[2,:] = [a,-a]
                        zg[3,:] = [0,a]
                        zg[4,:] = [0,0]
                        zg[5,:] = [0,-a]
                        zg[6,:] = [-a,a]
                        zg[7,:] = [-a,0]
                        zg[8,:] = [-a,-a]
                        wg =np.array([w1,w2,w1,w2,w4,w2,w1,w2,w1])
                    case 4:
                        Ng = 16
                        a = 0.861136311594053
                        b = 0.339981043584856
                        w1 = 0.1210029932856021
                        w2 = 0.2268518518518519
                        w3 = 0.4252933030106941
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a,a]
                        zg[1,:] = [a,b]
                        zg[2,:] = [a,-b]
                        zg[3,:] = [a,-a]
                        zg[4,:] = [b,a]
                        zg[5,:] = [b,b]
                        zg[6,:] = [b,-b]
                        zg[7,:] = [b,-a]
                        zg[8,:] = [-b,a]
                        zg[9,:] = [-b,b]
                        zg[10,:] = [-b,-b]
                        zg[11,:] = [-b,-a]
                        zg[12,:] = [-a,a]
                        zg[13,:] = [-a,b]
                        zg[14,:] = [-a,-b]
                        zg[15,:] = [-a,-a]
                        wg =np.array([w1,w2,w2,w1,w2,w3,w3,w2,w2,w3,w3,w2,w1,w2,w2,w1])
                    case 5:
                        Ng = 25
                        a = 0
                        b = 0.538469310105683
                        c = 0.906179845938663
                        W1 = 0.568888888888888
                        W2 = 0.478628670499366
                        W3 = 0.236926885056189
                        zg = np.zeros([Ng,2])
                        zg[0,:] = [a,a]
                        zg[1,:] = [c,c]
                        zg[2,:] = [c,b]
                        zg[3,:] = [c,a]
                        zg[4,:] = [b,c]
                        zg[5,:] = [b,b]
                        zg[6,:] = [b,a]
                        zg[7,:] = [a,c]
                        zg[8,:] = [a,b]
                        zg[9,:] = [-c,c]
                        zg[10,:] = [-c,b]
                        zg[11,:] = [-c,a]
                        zg[12,:] = [-b,c]
                        zg[13,:] = [-b,b]
                        zg[14,:] = [-b,a]
                        zg[15,:] = [c,-c]
                        zg[16,:] = [c,-b]
                        zg[17,:] = [b,-c]
                        zg[18,:] = [b,-b]
                        zg[19,:] = [a,-c]
                        zg[20,:] = [a,-b]
                        zg[21,:] = [-c,-c]
                        zg[22,:] = [-c,-b]
                        zg[23,:] = [-b,-c]
                        zg[24,:] = [-b,-b]
                        wg =np.array([W1*W1,W3*W3,W3*W2,W3*W1,W2*W3,W2*W2,W2*W1,W1*W3,W1*W2,W3*W3,W3*W2,W3*W1,W2*W3,W2*W2,W2*W1,W3*W3,W3*W2,W2*W3,W2*W2,W1*W3,W1*W2,W3*W3,W3*W2,W2*W3,W2*W2])
                
    return zg, wg, Ng

