#ifndef DARKVORTEX_CLASSIFIER_H
#define DARKVORTEX_CLASSIFIER_H

#include <string>

void train_classifier(const std::string& datacfg, const std::string& cfgfile, const std::string& weightfile, int* gpus, int ngpus, int clear);
void validate_classifier_single(const std::string& datacfg, const std::string& cfgfile, const std::string& weightfile);

#endif
