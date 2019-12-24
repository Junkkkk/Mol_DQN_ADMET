import sys
sys.path.insert(0,'/home/junyoung/workspace/ADMET/chemopy-1.0/src')
from pychem import PyChem2d
import numpy as np

class ADMET():
    def __init__(self, mol):
        self.descriptors = PyChem2d(mol)
        self.Descriptor = self.Get_Descriptor()

    def Get_Descriptor(self):

        basak = self.descriptors.GetBasak()
        # kappa = self.descriptors.GetKappa()
        top = self.descriptors.GetTopology()
        conv = self.descriptors.GetConnectivity()
        con = self.descriptors.GetConstitution()
        moran = self.descriptors.GetMoran()
        moe = self.descriptors.GetMOE()
        mol = self.descriptors.GetMolProperty()
        bcu = self.descriptors.GetBcut()
        est = self.descriptors.GetEstate()
        cha = self.descriptors.GetCharge()
        more = self.descriptors.GetMoreauBroto()
        ecfp4 = self.descriptors.GetMorganFingerprint()
        # gea = self.descriptors.GetGeary()

        logS_feature = [moran['MATSm2'], top['TIAC'], top['GMTIV'], basak['IC1'],
                        con['naro'], moran['MATSm1'], con['nsulph'], cha['Tpc'],
                        moe['slogPVSA7'], bcu['bcutp1'], con['AWeight'], cha['Tnc'],
                        moe['MRVSA9'],  bcu['bcutp3'], basak['IC0'], top['AW'], mol['Hy'],
                        bcu['bcutv10'], moe['MRVSA6'], con['PC6'], bcu['bcutm1'],
                        bcu['bcutm8'], moe['slogPVSA1'], top['IDET'],
                        conv['Chi10'], mol['TPSA'], con['Weight'], cha['Rnc'], con['naccr'],
                        bcu['bcutp5'], conv['Chiv4'], bcu['bcutm2'], conv['Chiv1'],
                        bcu['bcutm3'], conv['Chiv9'], con['ncarb'], bcu['bcutm4'],
                        moe['PEOEVSA5'], mol['LogP2'], mol['LogP']]

        caco_feature = [con['ncarb'], basak['IC0'], bcu['bcutp1'], bcu['bcutv10'],
                        top['GMTIV'], con['nsulph'], basak['CIC6'],bcu['bcutm12'],
                        est['S34'], bcu['bcutp8'], moe['slogPVSA2'], cha['QNmin'],
                        mol['LogP2'], bcu['bcutm1'], moe['EstateVSA9'], moe['slogPVSA1'],
                        top['Hatov'], top['J'], top['AW'], est['S7'], conv['dchi0'],
                        moe['MRVSA1'], mol['LogP'], cha['Tpc'], moe['PEOEVSA0'],
                        cha['Tnc'], est['S13'], mol['TPSA'], cha['QHss'], con['ndonr']]

        # ppb_feature = [con['ncarb'], con['naro'], top['GMTIV'], top['AW'], top['Geto'],
        #                top['Arto'], top['BertzCT'], top['J'], kappa['kappam3'], moran['MATSv1'],
        #                moran['MATSe1'], mol['Hy'], mol['LogP'], mol['LogP2'], mol['UI'],
        #                cha['QHss'], cha['LDI'], cha['QNss'], cha['Rpc'], cha['Mnc'],
        #                moe['PEOEVSA10'], moe['PEOEVSA0'], moe['PEOEVSA6'], moe['PEOEVSA5'],
        #                moe['PEOEVSA4'], moe['slogPVSA10'], moe['MRVSA6'], moe['slogPVSA0'],
        #                moe['slogPVSA1'], moe['slogPVSA5']]

        cyp_3a4_feature = ecfp4

        t_feature = [moran['MATSv5'], top['Gravto'], conv['Chiv3c'], moe['PEOEVSA7'], conv['knotp'],
                     bcu['bcutp3'], bcu['bcutm9'], moe['EstateVSA3'], moran['MATSp1'],
                     bcu['bcutp11'], moe['VSAEstate7'], basak['IC0'], mol['UI'], top['Geto'],
                     cha['QOmin'], basak['CIC0'], conv['dchi3'], moran['MATSp4'], bcu['bcutm4'],
                     top['Hatov'], moran['MATSe4'], basak['CIC6'], conv['Chiv4'], moe['EstateVSA9'],
                     moran['MATSv2'], con['nring'], bcu['bcute1'], moe['VSAEstate8'], moe['MRVSA9'],
                     moe['PEOEVSA6'], basak['SIC1'], bcu['bcutp8'], moran['MATSp6'], cha['QCss'],
                     top['J'], top['IDE'], basak['CIC2'], mol['Hy'], moe['MRVSA6'], con['naro'],
                     cha['SPP'], moe['EstateVSA7'], bcu['bcutv10'], est['S12'], mol['LogP2'],
                     bcu['bcutp2'], basak['CIC3'], est['S17'], mol['LogP'], bcu['bcutp1']]

        ld50_feature = [more['ATSm1'], more['ATSm2'], more['ATSm3'], more['ATSm4'], more['ATSm6'],
                        con['AWeight'], conv['Chi4c'], conv['Chiv3'], conv['Chiv4'], conv['Chiv4c'],
                        conv['Chiv4pc'], est['DS'], top['Gravto'], basak['IC0'], basak['IC1'],
                        moe['MRVSA9'], cha['QCmax'], cha['QNss'], cha['QOmin'], cha['Qmax'], est['S46'],
                        est['Smax45'], est['Smin'], est['Smin45'], moe['VSAEstate7'], con['Weight'],
                        bcu['bcutm1'],bcu['bcutm2'], bcu['bcutp1'], con['nhet'], con['nphos'], moe['slogPVSA11']]

        # hERG_feature = [con['ndb'], con['nsb'], con['ncarb'], con['nsulph'], con['naro'], con['ndonr'],
        #                 con['nhev'], con['naccr'], con['nta'], con['nring'], con['PC6'], top['GMTIV'],
        #                 top['AW'], top['Geto'], top['BertzCT'], top['J'], top['MZM2'], kappa['phi'],
        #                 kappa['kappa2'], moran['MATSv1'], moran['MATSv5'], moran['MATSe4'], moran['MATSe5'],
        #                 moran['MATSe6'], mol['TPSA'], mol['Hy'], mol['LogP'], mol['LogP2'], mol['UI'],
        #                 cha['QOss'], cha['SPP'], cha['LDI'], cha['Qass'], cha['QOmin'], cha['QNmax'], cha['Qmin'],
        #                 cha['Mnc'], moe['EstateVSA7'], moe['EstateVSA0'], moe['EstateVSA3'], moe['PEOEVSA0'],
        #                 moe['PEOEVSA6'], moe['MRVSA5'], moe['MRVSA4'], moe['MRVSA3'], moe['MRVSA6'], moe['slogPVSA1']]

        res = {'logS': logS_feature,
               'caco': caco_feature,
               'cyp_3a4': cyp_3a4_feature,
               't': t_feature,
               'ld50': ld50_feature
               }

        return res

    def Get_logS(self, model):
        return float(model.predict(self.Descriptor['logS']))

    def Get_caco(self, model):
        return float(model.predict(self.Descriptor['caco']))

    # def Get_ppb(self, model):
    #     return float(model.predict(self.Descriptor['ppb']))

    def Get_t(self, model):
        return float(model.predict(self.Descriptor['t']))

    def Get_cyp3a4(self, model):
        return float(model.predict_proba(self.Descriptor['cyp_3a4'])[0][0])

    def Get_ld50(self, model):
        return float(model.predict(self.Descriptor['ld50']))

    # def Get_hERG(self, model):
    #     return float(model.predict_proba(self.Descriptor['hERG'])[0][1])