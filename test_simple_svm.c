#include <stdio.h>
#include <stdlib.h>
#include "simple_svm.h"

#define NEASYSAMPLES (100)

int main(int argc, char **argv)
{
  size_t i_sample;
  int    ans, predict_label, correct;
  float *easy_sample[NEASYSAMPLES];
  int   easy_sample_label[NEASYSAMPLES];
  float test[2];
  SmpSVMHn svm;
  SmpSVMConfig svm_config;

  for (i_sample = 0; i_sample < NEASYSAMPLES; i_sample++) {
    easy_sample[i_sample] = (float *)malloc(sizeof(float) * 2);
  }

  /* サンプルのセット */
  /* 簡単な例 : 2次元空間 */
  for (i_sample = 0; i_sample < NEASYSAMPLES; ++i_sample) {
    test[0] = (float)rand() / RAND_MAX;
    test[1] = (float)rand() / RAND_MAX;
    easy_sample_label[i_sample] = (test[0] >= 0.5f) ? 1 : -1;
    easy_sample[i_sample][0] = test[0];
    easy_sample[i_sample][1] = test[1];
  }
  smpSVM_SetDefaultConfig(&svm_config);
  svm_config.soft_margin_type = SMPSVM_SOFTMARGIN_ONE_NORM;
  svm_config.soft_margin_const = 0.4f;
  svm = smpSVM_Create(&svm_config);
  smpSVM_SetSample(svm, (const float **)easy_sample, easy_sample_label, NEASYSAMPLES, 2);
  smpSVM_Learning(svm);

  correct = 0;
  for (float i = -1; i <= 1; i += 0.1) {
    for (float j = -1; j <= 1; j += 0.1) {
      test[0] = i; test[1] = j;
      ans = (test[0] >= 0.5f) ? 1 : -1;
      smpSVM_PredictLabel(svm, &predict_label, test, 2);
      printf("<TEST> [%f,%f] label : %d answer : %d \r\n", test[0], test[1], predict_label, ans);
      if (ans == predict_label) {
        correct++;
      }
    }
  }

  printf("Accuracy:%f [%%]", (float)correct*100/(21*21));

  return 0;
}
