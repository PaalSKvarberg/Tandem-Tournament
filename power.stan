data
{
  int pre_intervention_question_count;
  int post_intervention_question_count;
  int antall_team;
  int antall_medlemmer_per_team;
  
  real intervensjons_effekt_individ;
  real intervensjons_effekt_team;
  vector[2] question_parameters; //beta verdier for fordelingen de sanne sannsynlghetene trekkes fra
  vector[2] individual_parameters; //beta verdier for korrelasjonen mellom folks gjetning og sannhet
  vector[2] team_learning_parameters;
}
transformed data {
  int totalt_antall_deltagere = antall_team * antall_medlemmer_per_team;
  int<lower = 1, upper = antall_team> individual_to_team[totalt_antall_deltagere];
  int last_individual_treated;
  int last_team_treated;
  
  {
  int temp_count = 1;
  
  for (team_i in 1:antall_team){
    if (team_i < antall_team/2.0){
      last_team_treated = team_i;
      last_individual_treated = last_team_treated * antall_medlemmer_per_team;
    }
    
    for (member_i in 1:antall_medlemmer_per_team){
    individual_to_team[temp_count] = team_i;
    
    temp_count += 1;
      
    }
    
  }
    
  }
}
generated quantities {
  vector[pre_intervention_question_count] pre_questions_truth;
  vector[post_intervention_question_count] post_questions_truth;
  
  matrix[totalt_antall_deltagere, pre_intervention_question_count] pre_teamdebate_pre_intervention_responses;
  matrix[totalt_antall_deltagere, pre_intervention_question_count] post_teamdebate_pre_intervention_responses;

  matrix[totalt_antall_deltagere, post_intervention_question_count] pre_teamdebate_post_intervention_responses;
  matrix[totalt_antall_deltagere, post_intervention_question_count] post_teamdebate_post_intervention_responses;
  
  int pre_questions_outcome[pre_intervention_question_count];
  int post_questions_outcome[post_intervention_question_count];
  
  vector[totalt_antall_deltagere] individ_baseline; // Korrelasjonen mellom individers gjetning og sannheten
  vector[totalt_antall_deltagere] individ_postintervention; // Korrelasjonen mellom individers gjetning og sannheten
  vector[antall_team] team_learning;
  vector[antall_team] team_learning_postintervention;
  
  vector[totalt_antall_deltagere] pre_teamdebate_pre_intervention_brier;
  vector[totalt_antall_deltagere] post_teamdebate_pre_intervention_brier;

  vector[totalt_antall_deltagere] pre_teamdebate_post_intervention_brier;
  vector[totalt_antall_deltagere] post_teamdebate_post_intervention_brier;

  
  // Vi setter sanne verdier på spørsmålene - på logit skala
  for (i in 1:pre_intervention_question_count){
    pre_questions_truth[i] = logit(beta_rng(question_parameters[1], question_parameters[2]));
}
  for (i in 1:post_intervention_question_count){
    post_questions_truth[i] = logit(beta_rng(question_parameters[1], question_parameters[2]));
}

  // Vi setter folks dyktighet
  for (i in 1:totalt_antall_deltagere){
    individ_baseline[i] = logit(beta_rng(individual_parameters[1], individual_parameters[2]));
    individ_postintervention[i] = individ_baseline[i] + 
              ((i <= last_individual_treated)? intervensjons_effekt_individ : 0);
}

  // Vi setter hvordan team forbedrer deltagere
  for (i in 1:antall_team){
    team_learning[i] = normal_rng(team_learning_parameters[1],
                                  team_learning_parameters[2]);
    team_learning_postintervention[i] = team_learning[i] + 
          ((i <= last_team_treated)? intervensjons_effekt_team : 0);
  }
  
  // Vi trekker folks gjetninger før de diskuterer innad i team
  for (question_i in 1:pre_intervention_question_count){
    for (individual_i in 1:totalt_antall_deltagere){
      pre_teamdebate_pre_intervention_responses[individual_i, question_i] = pre_questions_truth[question_i] * 
      inv_logit(individ_baseline[individual_i]) +
            normal_rng(0, sqrt(1 - inv_logit(individ_baseline[individual_i])^2));
}
}
  for (question_i in 1:post_intervention_question_count){
    for (individual_i in 1:totalt_antall_deltagere){
      pre_teamdebate_post_intervention_responses[individual_i, question_i] = post_questions_truth[question_i] * 
      inv_logit(individ_postintervention[individual_i]) +
            normal_rng(0, sqrt(1 - inv_logit(individ_postintervention[individual_i])^2));
}
}
  // Vi beregner folks gjetninger etter at de har diskutert innad i team (før intervensjon)
  for (question_i in 1:pre_intervention_question_count){
    for (individual_i in 1:totalt_antall_deltagere){
      real temp_corr = inv_logit(individ_baseline[individual_i] +
                    team_learning[individual_to_team[individual_i]]);
                    
      post_teamdebate_pre_intervention_responses[individual_i, question_i] = 
      pre_questions_truth[question_i] * temp_corr +
            normal_rng(0, sqrt(1 - temp_corr^2));
}
}
  
  // Vi beregner folks gjetninger etter at de har diskutert innad i team (etter intervensjon)
  for (question_i in 1:post_intervention_question_count){
    for (individual_i in 1:totalt_antall_deltagere){
      real temp_corr = inv_logit(individ_postintervention[individual_i] +
                                    team_learning_postintervention[individual_to_team[individual_i]]);
      
      post_teamdebate_post_intervention_responses[individual_i, question_i] = 
      post_questions_truth[question_i] * temp_corr +
            normal_rng(0, sqrt(1 - temp_corr^2));
}
}
  
  
  // Vi trekker observerte utfall
  
  pre_questions_outcome = bernoulli_rng(inv_logit(pre_questions_truth));
  post_questions_outcome = bernoulli_rng(inv_logit(post_questions_truth));

  // Vi beregner Brier skår
  {
    real temp_brier_pre_pre;
    real temp_brier_pre_post;
    real temp_brier_post_pre;
    real temp_brier_post_post;

  for (individual_i in 1:totalt_antall_deltagere){
    temp_brier_pre_pre = 0;
    temp_brier_pre_post = 0;
    temp_brier_post_pre = 0;
    temp_brier_post_post = 0;

    for (question_i in 1:pre_intervention_question_count){
      temp_brier_pre_pre += (pre_questions_outcome[question_i] -
      inv_logit(pre_teamdebate_pre_intervention_responses[individual_i, question_i]))^2;

      temp_brier_post_pre += (pre_questions_outcome[question_i] -
       inv_logit(post_teamdebate_pre_intervention_responses[individual_i, question_i]))^2;


    }
    pre_teamdebate_pre_intervention_brier[individual_i] = temp_brier_pre_pre/(1.0 * pre_intervention_question_count);
    post_teamdebate_pre_intervention_brier[individual_i] = temp_brier_post_pre/(1.0 * pre_intervention_question_count);

    for (question_i in 1:post_intervention_question_count){
      temp_brier_pre_post += (post_questions_outcome[question_i] -
       inv_logit(pre_teamdebate_post_intervention_responses[individual_i, question_i]))^2;

      temp_brier_post_post += (post_questions_outcome[question_i] -
       inv_logit(post_teamdebate_post_intervention_responses[individual_i, question_i]))^2;

    }

    pre_teamdebate_post_intervention_brier[individual_i] = temp_brier_pre_post/(1.0 * post_intervention_question_count);
    post_teamdebate_post_intervention_brier[individual_i] = temp_brier_post_post/(1.0 * post_intervention_question_count);

}
  }

  

}

