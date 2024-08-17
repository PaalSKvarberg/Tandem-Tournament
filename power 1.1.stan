data
{
  int pre_intervention_question_count;
  int post_intervention_question_count;
  int antall_team;
  int antall_medlemmer_per_team;
  
  real intervensjons_effekt_individ;
  real intervensjons_effekt_individ_certainty;
  real intervensjons_effekt_team;
  real intervensjons_effekt_team_certainty;
  vector[2] question_parameters; //beta verdier for fordelingen de sanne sannsynlghetene trekkes fra
  vector[2] individual_parameters; //mean og sd for sensitivitet til faktisk sannsynlighet
  vector[2] individual_parameters_certainty; //mean og sd for presisjonsparameter (will use exp of draw to ensure positivity)
  vector[2] team_learning_parameters;
  vector[2] team_learning_parameters_certainty;
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
  row_vector[pre_intervention_question_count] pre_questions_truth;
  row_vector[post_intervention_question_count] post_questions_truth;
  
  matrix[totalt_antall_deltagere, pre_intervention_question_count] pre_teamdebate_pre_intervention_responses;
  matrix[totalt_antall_deltagere, pre_intervention_question_count] post_teamdebate_pre_intervention_responses;

  matrix[totalt_antall_deltagere, post_intervention_question_count] pre_teamdebate_post_intervention_responses;
  matrix[totalt_antall_deltagere, post_intervention_question_count] post_teamdebate_post_intervention_responses;
  
  int pre_questions_outcome[pre_intervention_question_count];
  int post_questions_outcome[post_intervention_question_count];
  
  matrix[totalt_antall_deltagere, 4] individual_beta; // Rounds in rows: pre intervention pre team, pre int post team, post int pre team, post int post team
  matrix[totalt_antall_deltagere, 4] individual_beta_certainty;

  vector[antall_team] team_learning;
  vector[antall_team] team_learning_certainty;
  vector[antall_team] team_learning_postintervention;
  vector[antall_team] team_learning_postintervention_certainty;
  
  vector[totalt_antall_deltagere] pre_teamdebate_pre_intervention_brier;
  vector[totalt_antall_deltagere] post_teamdebate_pre_intervention_brier;

  vector[totalt_antall_deltagere] pre_teamdebate_post_intervention_brier;
  vector[totalt_antall_deltagere] post_teamdebate_post_intervention_brier;

  
  // We draw the "best estimate possible" ("truth") for each question
  
  for (i in 1:pre_intervention_question_count){
    pre_questions_truth[i] = logit(beta_rng(question_parameters[1], question_parameters[2]));
}
  for (i in 1:post_intervention_question_count){
    post_questions_truth[i] = logit(beta_rng(question_parameters[1], question_parameters[2]));
}

  // Team effects
  for (i in 1:antall_team){
    team_learning[i] = beta_rng(team_learning_parameters[1],
                                  team_learning_parameters[2]);
    team_learning_postintervention[i] = inv_logit(logit(team_learning[i]) + 
          ((i <= last_team_treated)? intervensjons_effekt_team : 0));
    team_learning_certainty[i] = normal_rng(team_learning_parameters_certainty[1],
                                  team_learning_parameters_certainty[2]);
    team_learning_postintervention_certainty[i] = team_learning_certainty[i] + 
          ((i <= last_team_treated)? intervensjons_effekt_team_certainty : 0);
  }



  // Participant parameters
  for (i in 1:totalt_antall_deltagere){
    // Baseline - pre-team and pre-intervention
    individual_beta[i, 1] = normal_rng(individual_parameters[1], individual_parameters[2]);
    individual_beta_certainty[i, 1] = exp(normal_rng(individual_parameters_certainty[1], individual_parameters_certainty[2]));
    
    // Before intervention but after team
    
    individual_beta[i, 2] = individual_beta[i, 1] + (team_learning[individual_to_team[i]] * (1 - individual_beta[i, 1]));
    individual_beta_certainty[i, 2] = exp(log(individual_beta_certainty[i, 1]) + team_learning_certainty[individual_to_team[i]]);
    
    // After intervention before team
    
    individual_beta[i, 3] = individual_beta[i, 1] + 
              (((i <= last_individual_treated) ? intervensjons_effekt_individ : 0) * (1 - individual_beta[i,1]));
    individual_beta_certainty[i, 3] = exp(log(individual_beta_certainty[i, 1]) + 
              ((i <= last_individual_treated) ? intervensjons_effekt_individ_certainty : 0));
              
    // After intervention and after team
    
    individual_beta[i, 4] = individual_beta[i, 3] + (team_learning_postintervention[individual_to_team[i]] * (1- individual_beta[i, 3]));
    individual_beta_certainty[i, 4] = exp(log(individual_beta_certainty[i, 3]) + team_learning_postintervention_certainty[individual_to_team[i]]);
    
    
}

  
  // We draw participant guesses
  { // Pre-intervention
        matrix[totalt_antall_deltagere, pre_intervention_question_count] temp_expectation_pre_team =
           inv_logit(logit(0.5) + individual_beta[, 1] * (pre_questions_truth - logit(0.5)));
        matrix[totalt_antall_deltagere, pre_intervention_question_count] temp_expectation_post_team =
           inv_logit(logit(0.5) + individual_beta[, 2] * (pre_questions_truth - logit(0.5)));
        real temp_term;

       for (question_i in 1:pre_intervention_question_count){
         for (individual_i in 1:totalt_antall_deltagere){
           temp_term = temp_expectation_pre_team[individual_i, question_i];

           pre_teamdebate_pre_intervention_responses[individual_i, question_i] = beta_rng( temp_term *
              individual_beta_certainty[individual_i, 1],
              (1 - temp_term) * individual_beta_certainty[individual_i, 1]);
              
           temp_term = temp_expectation_post_team[individual_i, question_i];

           post_teamdebate_pre_intervention_responses[individual_i, question_i] = beta_rng( temp_term *
              individual_beta_certainty[individual_i, 2],
              (1 - temp_term) * individual_beta_certainty[individual_i, 2]);
              
        
      }
    }
  }
  
  { // Post-intervention
        matrix[totalt_antall_deltagere, post_intervention_question_count] temp_expectation_pre_team =
           inv_logit(logit(0.5) + individual_beta[, 3] * (post_questions_truth - logit(0.5)));
        matrix[totalt_antall_deltagere, post_intervention_question_count] temp_expectation_post_team =
           inv_logit(logit(0.5) + individual_beta[, 4] * (post_questions_truth - logit(0.5)));
        real temp_term;

       for (question_i in 1:post_intervention_question_count){
         for (individual_i in 1:totalt_antall_deltagere){
           temp_term = temp_expectation_pre_team[individual_i, question_i];

           pre_teamdebate_post_intervention_responses[individual_i, question_i] = beta_rng( temp_term *
              individual_beta_certainty[individual_i, 3],
              (1 - temp_term) * individual_beta_certainty[individual_i, 3]);

           temp_term = temp_expectation_post_team[individual_i, question_i];

           post_teamdebate_post_intervention_responses[individual_i, question_i] = beta_rng( temp_term *
              individual_beta_certainty[individual_i, 4],
              (1 - temp_term) * individual_beta_certainty[individual_i, 4]);


      }
    }
  }
    
  // Vi trekker observerte utfall
  
  pre_questions_outcome = bernoulli_rng(inv_logit(pre_questions_truth));
  post_questions_outcome = bernoulli_rng(inv_logit(post_questions_truth));

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
      pre_teamdebate_pre_intervention_responses[individual_i, question_i])^2;

      temp_brier_post_pre += (pre_questions_outcome[question_i] -
       post_teamdebate_pre_intervention_responses[individual_i, question_i])^2;

       // Implement alternative scoring by including both options

      temp_brier_pre_pre += ((pre_questions_outcome[question_i] == 0) -
      (1 - pre_teamdebate_pre_intervention_responses[individual_i, question_i]))^2;

      temp_brier_post_pre += ((pre_questions_outcome[question_i] == 0) -
       (1 - post_teamdebate_pre_intervention_responses[individual_i, question_i]))^2;


    }
    pre_teamdebate_pre_intervention_brier[individual_i] = temp_brier_pre_pre/(1.0 * pre_intervention_question_count);
    post_teamdebate_pre_intervention_brier[individual_i] = temp_brier_post_pre/(1.0 * pre_intervention_question_count);

    for (question_i in 1:post_intervention_question_count){
      temp_brier_pre_post += (post_questions_outcome[question_i] -
       pre_teamdebate_post_intervention_responses[individual_i, question_i])^2;

      temp_brier_post_post += (post_questions_outcome[question_i] -
       post_teamdebate_post_intervention_responses[individual_i, question_i])^2;

       // Implement alternative scoring by including both options


      temp_brier_pre_post += ((post_questions_outcome[question_i] == 0) -
       (1 -pre_teamdebate_post_intervention_responses[individual_i, question_i]))^2;

      temp_brier_post_post += ((post_questions_outcome[question_i] == 0)-
       (1- post_teamdebate_post_intervention_responses[individual_i, question_i]))^2;

    }

    pre_teamdebate_post_intervention_brier[individual_i] = temp_brier_pre_post/(1.0 * post_intervention_question_count);
    post_teamdebate_post_intervention_brier[individual_i] = temp_brier_post_post/(1.0 * post_intervention_question_count);

}
  }



}

