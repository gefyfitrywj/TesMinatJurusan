<template>
	<div class="container">
		<div class="text-4xl">
			You've Arrived Here on Error, boss.
		</div>
		<button
			class="font-bold button"
			@click="goBack">
			Back
		</button>
	</div>
</template>

<script>
export default {
	methods: {
		goBack() {
			this.$router.back();
		},
	},
};
</script>

  <style scoped>
  .container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
  }

  .text-4xl {
    text-align: center;
    margin-bottom: 20px;
  }

  .button {
    padding: 4px 6px;
    margin: 10px 0px;
    background: black;
    color: white;
  }
  </style>
