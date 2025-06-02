<template>
    <main class="flex flex-col bg-[#F9F9FF] min-h-screen justify-center items-center">
        <section id="hero" class="content-container flex justify-between w-full max-h-[359px] mt-[120px] mb-[87px]">
            <img src="assets/images/hero.webp" alt="hero image" class="object-contain order-2">
            <div class="flex flex-col gap-[32px] font-montserrat max-w-[590px] max-h-[296px] order-1">
                <div class="flex flex-col gap-[12px] tracking-[3%] text-wrap">
                    <h1 class="font-[700] text-[32px] leading-[60px]">Arahkan Masa Depanmu, Mulai Dari Sini</h1>
                    <p class="text-[20px] leading-[40px]">Jangan Salah Jurusan, Ikuti Tes Minat Agar Masa Depanmu Sukses
                    </p>
                </div>
                <NuxtLink to="/clause"
                    class="bg-[#FCBC1D] shadow-[0_4_49_0_#00000026] w-[181px] rounded-[8px] text-center text-white py-[12px] px-[24px]">Tes
                    Minat</NuxtLink>
            </div>
        </section>
        <section id="about" class="content-container flex justify-between w-full max-h-[360px] mt-[108px] mb-[220px]">
            <img src="assets/images/about.webp" alt="about image" class="object-contain">
            <div class="flex flex-col justify-center gap-[12px] font-montserrat max-w-[590px]">
                <h1 class="font-[700] text-[32px] leading-[60px]">Yuk, Cari Tahu Jurusan Kamu!</h1>
                <p class="text-[20px] leading-[40px]">Kamu akan mengerjakan soal yang menggambarkan dirimu, hasilnya
                    akan keluar
                    dan bisa kamu jadikan acuan untuk memilih jurusan</p>
            </div>
        </section>
        <section id="features" class="content-container flex flex-col gap-[112px] mb-[183px]">
            <h1 class="font-[700] text-[32px] leading-[60px] text-center">Tahukah Kamu?</h1>
            <div class="flex flex-row justify-center gap-[92px] flex-wrap text-[#4A5568]">
                <p class="border border-solid px-[16px] py-[26px] text-[14px] leading-[162%] max-w-[333px] text-center"
                    style="border-image-source: linear-gradient(225deg, #F76680 0%, #57007B 100%); border-image-slice: 1;">
                    87% Mahasiswa salah jurusan.
                    Banyak pelajar memilih jurusan tanpa tahu minat dan bakat mereka yang sebenarnya. Akibatnya? Kuliah
                    terasa
                    berat, dan karier jadi tidak sesuai harapan</p>
                <p class="border border-solid px-[16px] py-[26px] text-[14px] leading-[162%] max-w-[333px] text-center"
                    style="border-image-source: linear-gradient(225deg, #F76680 0%, #57007B 100%); border-image-slice: 1;">
                    Salah
                    jurusan bisa bikin kamu harus mengulang kuliah dari awal. Itu berarti waktu lebih lama dan biaya
                    pendidikan jadi lebih besar.</p>
                <p class="border border-solid px-[16px] py-[26px] text-[14px] leading-[162%] max-w-[333px] text-center"
                    style="border-image-source: linear-gradient(225deg, #F76680 0%, #57007B 100%); border-image-slice: 1;">
                    Jurusan
                    yang tidak sesuai bisa berdampak pada pekerjaan dan kebahagiaan di masa depan. Minat dan bakat yang
                    tidak tersalurkan akan membuatmu kurang maksimal.</p>
            </div>
        </section>
    </main>
</template>