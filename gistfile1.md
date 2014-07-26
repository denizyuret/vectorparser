### eParse deneylerini nasil calistiririm?
___________

Bu tutorialin amaci eParse deneylerinin tekrarlamak ya da yeni deneyler yapabilmek icin bir referans olusturmak.

Parser’in repository’si [burada](https://github.com/hsensoy/ai-parse). Burada oldukca aciklayici bir README mevcut. Bu dokumana devam etmeden mutlaka okunmasini oneririm.

#### 0- Kelime Vektorleri

Kelime vektorleri _@ai-ku servers:/ai/home/vcirik/embeddings_** altinda. Orada bir README
bulunmakta ama malesef henuz guncel degil. Her birinin metadatasini oradan inceleyebilirsiniz.

#### 1- Kelime Vektorlerinin ConLL datasina eklenmesi

Type based ve token based olmak uzere 2 cesit kelime vektorleri kullaniyoruz. Ilkinde bir kelime
tum corpusta tek bir vektorle represent edilirken, ikincisinde contexte bagli bir vektor kullaniyoruz. 

Type based vektorleri alip conLL’e eklemek oldukca basit. CONLL_DATA icerisinde train(00/wsj_0001.dp), development
(01/wsj_0101.dp) ve test (02/wsj_0201.dp) datalari olsun. 
      
      enrich_with_embeddings.py --delimiter " " --offset 1 --length 50 rcv1-50.embeddings CONLL_DATA CONLL_DATA_rcv1-50-type

Bu python scriptinin calistigi parserin READMEsinde var. Ama kisaca ozetliyelim, rcv1-50 kelime vektorleri icin
CONLL_DATA altindaki conll formatindaki dataya vektorleri ekliyor. Kelime vektorleri bosluk ile ayrilmis (" "),
50 dimension ve "kelime v_0 v_1... v_49" seklinde. offset kelimeden hemen sonra vektorun basladigini soyluyor.

Token based vektorleri cikarmak ise baska bir metodu gerektiriyor. O [surada](https://github.com/ai-ku/wvec) anlatiliyor.
Oncelikle conLL datasindan kelimeleri tokenized bir sekilde cumlelere cevirin.

        cat train.conll | cut -f2 | one-sentence-per-line.pl | gzip > train.tok.gz

Bunu development ve test icin de yaptiktan sonra, substitute distributionlari cikarmamiz gerek. wvec repositorysini
kullanarak bunu yaptiktan sonra {train|dev|test}.sub.gz yi kullanarak istedigimiz vektorleri icin token-based representation'i
cikartiyoruz.

        zcat train.sub.gz | concatSubs rcv1-50 rcv1-50 > embedded.train
        zcat dev.sub.gz | concatSubs rcv1-50 rcv1-50 > embedded.dev
        zcat test.sub.gz | concatSubs rcv1-50 rcv1-50  > embedded.test
        
burada hem context icin hem word type icin rcv1-50 kelime vektorlerini kullandik. Ama farkli kelime vektorlerini birlestirebiliriz(cw + mikolov, scode + hlbl etc.). _enrich_with_embeddings.py_** bizden tek bir dosya istedigi icin bu ic dosyayi birlestiriyoruz.
        
        cat embedded.train embedded.dev embedded.test | tr '\t' ' ' | sed 's/ $$//' > embedded.all

Artik embedded conll corpusu yaratabiliriz.

        enrich_with_embeddings.py --token --delimiter " " --offset 1 --length 100 CONLL_DATA CONLL_DATA_rcv1-50-token

#### 2- Parser'i Egitmek

Once parser'i derliyoruz. Derlemeden once icc derleyicisini devreye sokuyoruz.

      source /ai/opt/intel/bin/iccvars.sh intel64

Sonra derliyoruz.

      make CONF=Release-icc-Linux  

bu _ai-parse/dist/Release-icc-Linux/Intel_1-Linux-x86_ altinda ai-parse executable'ini birakiyor. 
Parseri egitmek icin cok sayida parametre var. Bunlarin hepsi repository'de aciklanmis durumda.

        ai-parse-icc -o rcv1-token-kernel -p CONLL_DATA_rcv1-50-token -s optimize -t 0 -d 1 -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl_lbf_rbf_root_dir -l 100 -k POLYNOMIAL -x LINEAR -c 15 -m 5000 2> logs/rcv1-token.log
        
Burada;
- -o : model ismi. Egitme bittiginde rcv1-token-kernel.model dosyasi cikacak.
- -p : kelime vektoru eklenmis conLL datasi. Onceki ornekteki data CONLL_DATA_rcv1-50-token kullanilmis
- -s : parser'in egitme/parse etme komutu. optimize yani egitme secilmis.
- -t : training corpus section. 0.section secilmis. daha detayli bilgi icin repository'e bakiniz 
- -d : development corpus section. 1.section secilmis.
- -e : feature pattern. daha detayli bilgi icin repository'e bakiniz
- -l : kelime vektorlerinin boyutu
- -k : kernel tipi.
- -x : daha detayli bilgi icin repository'e bakiniz
- -c : paralel core sayisi
- -m : training datasindan kac cumle ile egitmenin yapilacagi

Standard error logs klasoru altindaki rcv1-token.log dosyasina aktarilmis.

#### 2- Egitilmis Parser Modelini kullanmak

Parser modeli ciktiktan sonra test icin kullanilmasi ise cok daha basit.

    ai-parse-icc -o rcv1-token-kernel -p CONLL_DATA_rcv1-50-token -s parse -t 2 -e p-1v_p0v_p1v_c-1v_c0v_c1v_tl_lbf_rbf_root_dir -l 100 -k POLYNOMIAL -x LINEAR -c 15 2> logs/rcv1-token.test.log
    
Kullanilan opsiyonlar hemen hemen ayni sadece -t ile test section verip -s ile parser'in stage'ini belirliyoruz.