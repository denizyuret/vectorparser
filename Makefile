all:

# For conll2007 compatible conversion do not use the vadas np corrections.
# Apply pennconverter directly to original ptb files.
# The original options -conjAsHead -prepAsHead has been superseded by -conll2007.
# Cannot get identical files, possibly due to bug fixes in pennconverter.
# Difference of 220/5003 heads are different conll07 test.

%.dp: %.mrg
	java -jar pennconverter.jar -conll2007 < $< > $@



parser_py_train: wsj_0001.dp
	awk '{if(NF==0){print $$0}else{print $$2, $$4, $$7-1, $$8}}' $< > $@

parser_py_gold: wsj_0101.dp
	awk '{if(NF==0){print $$0}else{print $$2, $$4, $$7-1, $$8}}' $< > $@

parser_py_test: wsj_0101.dp
	cut -f2,4 $< | perl -pe 'if(/\S/){s/\t/\//;s/\n/ /;}' > $@

parser_py_model: parser_py_train parser_py_test parser_py_gold
	python parser.py $@ $^

clean:
	-rm parser_py_train parser_py_gold parser_py_test parser.pickle tagger.pickle
	-rm -rf parser_py_model
