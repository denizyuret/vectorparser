#!/usr/bin/perl -w
use strict;
my %cache;
while(<>) {
    if (/^(\d+\.\d+)\t(\d+\.\d+)\t(\d+)\t(\d+\.\d+)\t(\S+)$/) {
	my $avg = $2;
	my $feats = $5;
	if (not defined $cache{$feats}) {
	    $cache{$feats} = $avg;
	    print "$avg\t$feats\n";
	}
	if ($avg ne $cache{$feats}) {
	    die "$feats: $cache{$feats} -> $avg";
	}
    }
}
