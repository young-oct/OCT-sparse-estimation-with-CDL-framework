# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:06:14 2019

@author: SN-593
"""
import yaml

class OssiviewDataHeader:
    """Reads data from a binary file into a dictonary defining the meta data in teh contained file"""
    def __init__(self, filePath):
        self.filePath = filePath
        self.headerLen = self.getHeaderLength()
        self.fullLen = self.headerLen + len(str(self.headerLen))
        self.header = self.getHeader()[-self.headerLen:]
        self.metaData = self.parse()
                
    def RepresentsInt(self, s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
        
    def getHeaderLength(self):
        headerLen = ""
        with open(self.filePath,'rb') as fileobj:
            while True:
                ch = fileobj.read(1).decode('ascii')
                if(self.RepresentsInt(ch)):
                    headerLen = headerLen + ch
                else:
                    break
        return int(headerLen)
    
    def getHeader(self):
        header = ""
        with open(self.filePath,'rb') as f:
          for i in range(self.fullLen):
            c = f.read(1).decode('ascii')
            if not c:
                raise Exception ("Header exceeds file length - file corrupted")
            else:
                header = header + c
        return header
    
    def parse(self):
        return yaml.load(self.header,Loader=yaml.SafeLoader)
                   

if __name__ == "__main__":
    file = r'Data\testOutput.bin'
    Header = OssiviewDataHeader(file)
           
