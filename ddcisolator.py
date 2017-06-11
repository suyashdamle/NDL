import re
import csv
import os

field1='title'
field2='abstract'                     # Keep the possibly-empty field here
path='sorted_title'
count=0

def main():
     csv.field_size_limit(1024*1024*512)
     '''
     isolator("IEEE_title1.csv")
     '''
     for file in os.listdir(path):
        isolator(file)
     print count

def isolator(filename):
     print filename
     inputF=open(path+"/"+filename,"rb")
     filename=filename.strip('.csv')
     reader=csv.reader(inputF)
     outputF=open("abs_title_primary/"+filename+"_a_1.csv","wb")
     writer=csv.writer(outputF)
     field2Idx=-1
     writer.writerow(['ddc',field1,field2])

     newRow=[]
     header=next(reader)
     for item in range(0,len(header)):
          if 'id'== header[item]:
               idIdx=item
          if 'ddc' in header[item]:
               ddcIdx=item
          if field1 in header[item]:
               field1Idx=item
          if field2 in header[item]:
               field2Idx=item
     print field2Idx
     for row in reader:
          flag=0
          #idData=row[idIdx]
          ddcData=row[ddcIdx]
          field1Data=row[field1Idx]
          if(field2Idx>=0):
               field2Data=row[field2Idx]
          else:
               field2Data=''
          global count
          ddcList=re.findall("\d\d\d::",ddcData) #\d*[\.]?\d*:: to check for decimal-containing classifications
          ddc_1=int(ddcList[0].strip(":"))/100
          writer.writerow([ddc_1,field1Data,field2Data])
          for item in ddcList:
               number=int(item.strip(":"))/100
               if not number==ddc_1:            #checking if ddc is ambiguous
                    writer.writerow([number,field1Data,field2Data])
          count+=1

     outputF.close()
                
                
main()    
    
        
        
    
